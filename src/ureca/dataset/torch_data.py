"""Stage 2 (dataset): turning waveforms into model input for PyTorch.

    x = preprocess(X[i], crop=(-10, 50), normalize="peak")
    ds = WaveformDataset(X, files["y"], train, crop=(-10, 50), normalize="peak")
    sampler = balanced_sampler(files["y"].values[train])
"""
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from .loading import CLASSES, FS, TRIGGER

LOG_EPS = 1e-3           # volts; "log" normalisation is linear below this, logarithmic above


def preprocess(x: np.ndarray, crop=None, normalize="none", scale=1.0) -> np.ndarray:
    """One waveform (or a batch in the last axis) -> model input.

    crop:      (start_us, end_us) relative to the trigger, or None for the full window
    normalize: "none"   - volts, baseline removed
               "global" - divided by a fixed `scale` shared by all files
               "peak"   - each file divided by its own peak, so only shape is left
               "log"    - sign(x) * log(1 + |x| / LOG_EPS), scaled so 1 V -> 1. Keeps
                          amplitude but compresses it, so small ringing stays visible
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
    elif normalize == "log":
        x = np.sign(x) * np.log1p(np.abs(x) / LOG_EPS) / np.log1p(1 / LOG_EPS)
    elif normalize != "none":
        raise ValueError(f"unknown normalize: {normalize}")
    return x.astype(np.float32)


class WaveformDataset(Dataset):
    """Item i -> (tensor of shape [1, length], class id).

    The leading 1 is the channel dimension that 1-D CNN layers expect.
    augment: optional function (raw waveform, rng) -> waveform, applied before
    preprocessing. Use it for training data only.
    """

    def __init__(self, X, y, rows, crop=None, normalize="none", scale=1.0, augment=None, seed=0):
        self.X = X
        self.y = np.asarray(y)
        self.rows = np.asarray(rows)
        self.kw = dict(crop=crop, normalize=normalize, scale=scale)
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        x = self.X[r]
        if self.augment is not None:
            x = self.augment(x, self.rng)
        x = preprocess(x, **self.kw)
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


def session_balanced_sampler(y, sessions, seed=0) -> WeightedRandomSampler:
    """Like balanced_sampler, but inside each class every recording session is
    drawn equally often, so one 60-file session does not drown out a 16-file one."""
    y, sessions = np.asarray(y), np.asarray(sessions)
    _, s_idx, s_size = np.unique(sessions, return_inverse=True, return_counts=True)
    n_sess = {c: len(np.unique(sessions[y == c])) for c in np.unique(y)}
    w = 1.0 / (s_size[s_idx] * np.array([n_sess[c] for c in y]))
    g = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(torch.tensor(w, dtype=torch.double), len(y), replacement=True, generator=g)
