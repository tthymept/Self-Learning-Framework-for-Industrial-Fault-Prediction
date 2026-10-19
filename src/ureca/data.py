"""Loading waveforms for models: file index, array cache, PyTorch Dataset.

    files = load_index()                 # one row per file (from the database)
    X = load_waveforms(files)            # (n_files, 16384) float32, cached
    train, val = fold_split(files, k)    # row positions for fold k
    ds = WaveformDataset(X, files["y"], train, crop=(-10, 50), normalize="peak")
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from .config import ensure_dirs
from .db import connect

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


def preprocess(x: np.ndarray, crop=None, normalize="none", scale=1.0) -> np.ndarray:
    """One waveform (or a batch in the last axis) -> model input.

    crop:      (start_us, end_us) relative to the trigger, or None for the full window
    normalize: "none"   - volts, baseline removed
               "global" - divided by a fixed `scale` shared by all files
               "peak"   - each file divided by its own peak, so only shape is left
    """
    x = x - np.median(x, axis=-1, keepdims=True)
    if crop is not None:
        a = TRIGGER + int(round(crop[0] * FS / 1e6))
        b = TRIGGER + int(round(crop[1] * FS / 1e6))
        x = x[..., a:b]
    if normalize == "global":
        x = x / scale
    elif normalize == "peak":
        x = x / np.abs(x).max(axis=-1, keepdims=True)
    elif normalize != "none":
        raise ValueError(f"unknown normalize: {normalize}")
    return x.astype(np.float32)


class WaveformDataset(Dataset):
    """Item i -> (tensor of shape [1, length], class id).

    The leading 1 is the channel dimension that 1-D CNN layers expect.
    """

    def __init__(self, X, y, rows, crop=None, normalize="none", scale=1.0):
        self.X = X
        self.y = np.asarray(y)
        self.rows = np.asarray(rows)
        self.kw = dict(crop=crop, normalize=normalize, scale=scale)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        x = preprocess(self.X[r], **self.kw)
        return torch.from_numpy(x)[None, :], int(self.y[r])


def class_weights(y, n_classes=len(CLASSES)) -> torch.Tensor:
    """Inverse-frequency weights (mean 1) for a weighted loss."""
    counts = np.bincount(y, minlength=n_classes).astype(float)
    w = counts.sum() / (n_classes * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32)


def balanced_sampler(y, seed=0) -> WeightedRandomSampler:
    """Draws files so every class appears equally often on average."""
    y = np.asarray(y)
    counts = np.bincount(y)
    w = 1.0 / counts[y]
    g = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(torch.tensor(w, dtype=torch.double), len(y), replacement=True, generator=g)
