"""Stage 2 (dataset): loading the file index and waveforms from storage.

    files = load_index()                 # one row per file (from the database)
    X = load_waveforms(files)            # (n_files, 16384) float32, cached
    train, val = fold_split(files, k)    # row positions for fold k

Preprocessing and PyTorch datasets are in torch_data.py.
"""
import numpy as np
import pandas as pd

from ..config import ensure_dirs
from ..db import connect

CLASSES = ["partial_discharge", "switching_noise", "transient_noise", "spark_lighter"]
FS = 125e6               # samples per second
TRIGGER = 8192           # sample index of the trigger (centre of the window)
CACHE_NAME = "waveforms.npz"


def load_index(conn=None) -> pd.DataFrame:
    """Labelled, valid files with their fold. Adds y (class id) and ts."""
    conn = conn or connect()
    files = pd.read_sql(
        "SELECT file_id, raw_path, label, timestamp, sha256, session_id, fold "
        "FROM files WHERE status = 'ok' AND label IS NOT NULL ORDER BY file_id",
        conn,
    )
    files["y"] = files["label"].map(CLASSES.index)
    files["ts"] = pd.to_datetime(files["timestamp"])
    return files


def load_waveforms(files: pd.DataFrame, verbose=True) -> np.ndarray:
    """Waveforms in the row order of `files`, as float32.

    Reading 16,384-line text files is slow, so arrays are cached in
    cache/waveforms.npz, keyed by each file's sha256. Files missing from the
    cache are read from processed/ and the cache is updated.
    """
    paths = ensure_dirs()
    cache = paths["cache"] / CACHE_NAME
    known = {}
    if cache.exists():
        with np.load(cache) as z:
            known = dict(zip(z["sha256"], z["X"]))

    missing = [i for i, h in enumerate(files["sha256"]) if h not in known]
    for i in missing:
        row = files.iloc[i]
        known[row["sha256"]] = np.loadtxt(paths["processed"] / row["raw_path"], dtype=np.float32)
    if missing:
        np.savez(cache, sha256=np.array(list(known)), X=np.stack(list(known.values())))
    if verbose:
        print(f"{len(files)} waveforms: {len(files) - len(missing)} from cache, {len(missing)} read from text")
    return np.stack([known[h] for h in files["sha256"]])


def fold_split(files: pd.DataFrame, k: int):
    """Row positions (train, val) when fold k is held out for validation."""
    val = np.flatnonzero(files["fold"].values == k)
    train = np.flatnonzero(files["fold"].values != k)
    return train, val
