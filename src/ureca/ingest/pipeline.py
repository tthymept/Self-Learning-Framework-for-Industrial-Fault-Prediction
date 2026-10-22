"""Stage 1 (ingest): validate dropped .txt files, copy them to processed/, register them in the database.

    uv run python -m ureca.ingest                    all batches in drop/
    uv run python -m ureca.ingest "<batch name>" -y  one batch, no prompt
"""
import hashlib
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
import zipfile

import numpy as np

from ..config import ensure_dirs
from ..db import local_copy
from ..dataset.folds import assign_sessions_and_folds
from .labels import resolve_label

N_SAMPLES = 16384
MAX_ABS = 1.5  # volts; sanity limit (spark files clip at ~1.0)
TS_RE = re.compile(r"(\d{14})")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_waveform(path: Path) -> np.ndarray:
    try:
        x = np.loadtxt(path, dtype=np.float64)
    except Exception as e:
        raise ValueError(f"unreadable: {e}")
    if x.ndim != 1 or x.size != N_SAMPLES:
        raise ValueError(f"expected {N_SAMPLES} values, got {x.size}")
    if not np.isfinite(x).all():
        raise ValueError("NaN or inf values")
    if np.abs(x).max() > MAX_ABS:
        raise ValueError(f"amplitude beyond ±{MAX_ABS} V")
    return x


def process_file(conn, paths, batch_id, label, f: Path):
    """Return None if ingested, else a rejection reason."""
    try:
        x = load_waveform(f)
    except ValueError as e:
        return str(e)

    m = TS_RE.search(f.stem)
    if not m:
        return "no 14-digit timestamp in filename"
    try:
        ts = datetime.strptime(m.group(1), "%Y%m%d%H%M%S").isoformat()
    except ValueError:
        return "unparseable timestamp"

    digest = sha256_of(f)
    if conn.execute("SELECT 1 FROM files WHERE sha256=?", (digest,)).fetchone():
        return "duplicate"

    dest = paths["processed"] / label / f.name
    if dest.exists() and sha256_of(dest) != digest:
        return "name collision with different content"

    # order: copy -> register -> delete original (safe to re-run after a crash)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(f, dest)
    conn.execute(
        """INSERT INTO files
           (batch_id, raw_path, label, timestamp, sha256, n_samples, peak_value)
           VALUES (?,?,?,?,?,?,?)""",
        (
            batch_id,
            dest.relative_to(paths["processed"]).as_posix(),
            label,
            ts,
            digest,
            int(x.size),
            float(np.max(np.abs(x - np.median(x)))),
        ),
    )
    conn.commit()
    f.unlink()
    return None


def reject(conn, paths, batch_id, batch_name, label, f: Path, reason):
    dest = paths["rejected"] / batch_name / label / f.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(f, dest)
    conn.execute(
        "INSERT INTO rejections (batch_id, filename, reason, rejected_at) VALUES (?,?,?,?)",
        (batch_id, f"{label}/{f.name}", reason, datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    f.unlink()



def unzip_to_drop(zip_path: Path, drop: Path) -> Path:
    """Extract zip into drop/<zipname>/ ; flatten a single wrapper folder if present."""
    name = re.sub(r"\s*\(\d+\)$", "", zip_path.stem)  # strip browser '(1)' suffix
    target = drop / name
    if target.exists():
        return target  # already extracted (e.g. a skipped batch being re-run)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(target)
    # if the zip held one wrapper folder containing the class folders, step into it
    subs = [p for p in target.iterdir() if p.is_dir() and p.name != "__MACOSX"]
    if len(subs) == 1 and not list(target.glob("*.txt")) and any(
        q.is_dir() for q in subs[0].iterdir()
    ):
        inner = subs[0]
        for item in inner.iterdir():
            shutil.move(str(item), target / item.name)
        inner.rmdir()
    return target


def ingest_batch(conn, paths, batch_dir: Path):
    cls_dirs = [
        p for p in sorted(batch_dir.iterdir())
        if p.is_dir() and p.name != "__MACOSX"
    ]

    unknown = [p.name for p in cls_dirs if resolve_label(p.name) is None]
    if unknown:
        print(
            f"[{batch_dir.name}] SKIPPED, unrecognised class folders: {unknown}. "
            "Rename them or add keywords in ureca/ingest/labels.py"
        )
        return None

    if not any(d.glob("*.txt") for d in cls_dirs):
        print(f"[{batch_dir.name}] no .txt files, skipped")
        return None

    cur = conn.execute(
        "INSERT INTO batches (name, ingested_at) VALUES (?,?)",
        (batch_dir.name, datetime.now().isoformat(timespec="seconds")),
    )
    batch_id = cur.lastrowid
    conn.commit()

    n_ok = n_rej = 0
    for cls_dir in cls_dirs:
        label = resolve_label(cls_dir.name)
        for f in sorted(cls_dir.glob("*.txt")):
            reason = process_file(conn, paths, batch_id, label, f)
            if reason is None:
                n_ok += 1
            else:
                reject(conn, paths, batch_id, batch_dir.name, label, f, reason)
                n_rej += 1

    conn.execute(
        "UPDATE batches SET n_ok=?, n_rejected=? WHERE batch_id=?",
        (n_ok, n_rej, batch_id),
    )
    conn.commit()
    print(f"[{batch_dir.name}] ok={n_ok} rejected={n_rej}")

    # best-effort cleanup of empty folders (OneDrive may lock them briefly)
    try:
        for d in sorted(batch_dir.rglob("*"), reverse=True):
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
        if batch_dir.exists() and not any(batch_dir.iterdir()):
            batch_dir.rmdir()
    except OSError:
        print(f"[{batch_dir.name}] couldn't remove empty folders; delete them manually")
    return batch_id


def summarize(batches):
    for b in batches:
        print(f"\n{b.name}")
        for d in sorted(p for p in b.iterdir() if p.is_dir() and p.name != "__MACOSX"):
            n = len(list(d.glob("*.txt")))
            print(f"  {d.name} -> {resolve_label(d.name) or 'UNRECOGNISED'}: {n} files")
    print()


def main(names=None, assume_yes=False):
    paths = ensure_dirs()
    drop = paths["drop"]

    zips = sorted(drop.glob("*.zip"))
    for z in zips:
        unzip_to_drop(z, drop)

    batches = (
        [drop / n for n in names]
        if names
        else sorted(p for p in drop.iterdir() if p.is_dir())
    )
    if not batches:
        print("Nothing in drop/")
        return

    summarize(batches)
    if not assume_yes:
        if input("Process these? [y/N] ").strip().lower() != "y":
            print("Cancelled. Files left in drop/.")
            return

    with local_copy() as conn:
        for b in batches:
            ingest_batch(conn, paths, b)
        print(conn.execute(
            "SELECT label, COUNT(*) FROM files GROUP BY label"
        ).fetchall())
        print(conn.execute(
            "SELECT reason, COUNT(*) FROM rejections GROUP BY reason"
        ).fetchall())
        assign_sessions_and_folds(conn)

    for z in zips:
        name = re.sub(r"\s*\(\d+\)$", "", z.stem)
        if not (drop / name).exists():
            z.unlink()

