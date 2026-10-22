"""Stage 2 (dataset): recording sessions and cross-validation folds.

Files of the same class recorded within SESSION_GAP_S seconds of each other
belong to one session (they are near-copies of each other). Whole sessions
are assigned to folds, so a session is never split between training and
validation.

Fold assignment is stable: sessions that already have a fold keep it, and new
sessions go to the fold that has the fewest files of that class.

    uv run python -m ureca.dataset.folds            assign and print a summary
    uv run python -m ureca.dataset.folds --dry-run  print what would be assigned
"""
from collections import Counter

import pandas as pd

from ..db import connect, local_copy

N_FOLDS = 3          # limited by switching noise, which has only 3 sessions
SESSION_GAP_S = 60   # files closer than this (same class) share a session


def find_sessions(files: pd.DataFrame) -> pd.Series:
    """Session id per file: '<label>@<first timestamp of the session>'."""
    out = pd.Series(index=files.index, dtype=object)
    for label, g in files.groupby("label"):
        g = g.sort_values("ts")
        new = g["ts"].diff().dt.total_seconds().fillna(float("inf")) > SESSION_GAP_S
        start = g["ts"].where(new).ffill()
        out[g.index] = label + "@" + start.dt.strftime("%Y-%m-%dT%H:%M:%S")
    return out


def plan(files: pd.DataFrame) -> pd.DataFrame:
    """Return files with new session_id and fold columns."""
    files = files.copy()
    files["ts"] = pd.to_datetime(files["timestamp"])
    files["session_id"] = find_sessions(files)
    old_fold = files["fold"]
    files["fold"] = pd.NA

    sessions = (
        files.groupby("session_id")
        .agg(label=("label", "first"), n=("file_id", "size"), start=("ts", "min"))
        .sort_values(["n", "start"], ascending=[False, True])
    )
    load = Counter()   # (label, fold) -> files assigned so far
    total = Counter()  # fold -> files assigned so far
    pending = []

    # sessions that already have a fold keep it (majority of their files)
    for sid, s in sessions.iterrows():
        prev = old_fold[files["session_id"] == sid].dropna()
        if len(prev):
            f = int(prev.mode().iloc[0])
            files.loc[files["session_id"] == sid, "fold"] = f
            load[s.label, f] += s.n
            total[f] += s.n
        else:
            pending.append((sid, s))

    # new sessions, largest first, go to the emptiest fold for their class
    for sid, s in pending:
        f = min(range(N_FOLDS), key=lambda k: (load[s.label, k], total[k], k))
        files.loc[files["session_id"] == sid, "fold"] = f
        load[s.label, f] += s.n
        total[f] += s.n

    files["fold"] = files["fold"].astype(int)
    return files


def summary(files: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab(
        [files["label"]],
        files["fold"].map(lambda f: f"fold {f}"),
        values=files["session_id"],
        aggfunc=lambda s: f"{len(s)} files / {s.nunique()} sess",
    ).fillna("-")


def assign_sessions_and_folds(conn, dry_run=False) -> pd.DataFrame:
    has_fold = any(r[1] == "fold" for r in conn.execute("PRAGMA table_info(files)"))
    files = pd.read_sql(
        f"SELECT file_id, label, timestamp, {'fold' if has_fold else 'NULL AS fold'} "
        "FROM files WHERE status = 'ok' AND label IS NOT NULL",
        conn,
    )
    files = plan(files)
    if not dry_run:
        conn.executemany(
            "UPDATE files SET session_id = ?, fold = ? WHERE file_id = ?",
            files[["session_id", "fold", "file_id"]].astype(object).values.tolist(),
        )
        conn.commit()

    print(summary(files).to_string())
    missing = [
        (lab, f) for lab in files["label"].unique() for f in range(N_FOLDS)
        if not ((files["label"] == lab) & (files["fold"] == f)).any()
    ]
    if missing:
        print(f"Warning: some folds have no files of a class: {missing}")
    return files


def main(dry_run=False):
    if dry_run:
        with connect() as conn:
            assign_sessions_and_folds(conn, dry_run=True)
    else:
        with local_copy() as conn:
            assign_sessions_and_folds(conn)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print only, do not write")
    main(ap.parse_args().dry_run)
