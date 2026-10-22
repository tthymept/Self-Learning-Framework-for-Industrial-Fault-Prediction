"""Stage 3 (models): small 1-D CNN on the raw waveform, trained and scored with the session folds.

    cfg = CNNConfig(sampler="session", augment="light")
    P = cv_cnn(X, files, cfg, seeds=range(10))  # (n_seeds, n_files, n_classes)
    pred = P.mean(axis=0).argmax(axis=1)        # seed ensemble, out-of-fold

Every file's probabilities come from a model that never saw its session.
Training length is fixed (no early stopping): stopping on the validation fold
would let the validation files choose the model.
"""
import hashlib
import json
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..config import ensure_dirs
from ..dataset.folds import N_FOLDS
from ..dataset.loading import CLASSES, fold_split
from ..dataset.torch_data import (WaveformDataset, balanced_sampler, preprocess,
                                  session_balanced_sampler)
from .features import AMPLITUDE, SHAPE, compute_features

# Results change slightly with the number of CPU threads (floating-point sums
# are added in a different order), so it is fixed for reproducible runs.
THREADS = 4


@dataclass
class CNNConfig:
    crop: tuple | None = (-10, 50)   # µs around the trigger, None = full 131 µs window
    normalize: str = "global"        # see data.preprocess
    augment: str = "none"            # "none", "light" (shift + noise) or "strong" (+ gain, polarity)
    sampler: str = "class"           # "class": balanced classes; "session": also balanced sessions
    label_smoothing: float = 0.0
    extra: str = "none"              # numbers fed to the last layer next to the CNN features:
                                     # "amplitude" (log peak, log rms, clipped) or "features" (all 14)
    width: int = 16                  # channels in the first block (doubles twice)
    kernel: int = 7
    n_blocks: int = 5
    dropout: float = 0.3
    epochs: int = 40
    batch_size: int = 32
    lr: float = 3e-3
    weight_decay: float = 1e-2

    def to_dict(self):
        return asdict(self)

    def changes(self):
        """Only the settings that differ from the defaults."""
        default = CNNConfig().to_dict()
        return {k: v for k, v in self.to_dict().items() if v != default[k]}


class SmallCNN(nn.Module):
    """n_blocks x (conv -> batch norm -> ReLU -> max-pool /4), then global
    average + max pooling over time and one linear layer. `n_extra` numbers per
    file (e.g. amplitude) can join the pooled features before the linear layer."""

    def __init__(self, n_classes=len(CLASSES), width=16, kernel=7, n_blocks=5, dropout=0.3, n_extra=0):
        super().__init__()
        layers, c_in = [], 1
        for b in range(n_blocks):
            c_out = width * 2 ** min(b, 2)
            layers += [nn.Conv1d(c_in, c_out, kernel, padding=kernel // 2, bias=False),
                       nn.BatchNorm1d(c_out), nn.ReLU(), nn.MaxPool1d(4)]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * c_in + n_extra, n_classes))

    def forward(self, x, extra):
        h = self.features(x)
        return self.head(torch.cat([h.mean(-1), h.amax(-1), extra], dim=1))


def augment_light(x, rng):
    """Changes that should not change the class: shift up to ±1 µs and extra
    noise (0.5 mV, about half the PD noise floor)."""
    x = np.roll(x, rng.integers(-125, 126))
    return x + rng.normal(0, 5e-4, x.shape)


def augment_strong(x, rng):
    """augment_light plus gain ±25% and a random polarity flip."""
    x = augment_light(x, rng) * np.exp(rng.uniform(np.log(0.8), np.log(1.25)))
    return -x if rng.random() < 0.5 else x


AUGMENT = {"none": None, "light": augment_light, "strong": augment_strong}


def global_scale(X, rows):
    """Median peak of the training files (volts), for normalize="global"."""
    D = X[rows] - np.median(X[rows], axis=1, keepdims=True)
    return float(np.median(np.abs(D).max(axis=1)))


def extra_inputs(X, kind):
    """Per-file numbers for the last layer, shape (n_files, n):
    "amplitude": log10 peak, log10 rms, clipped share
    "features":  log of all 14 baseline features (features.py)"""
    if kind == "none":
        return np.zeros((len(X), 0))
    if kind == "amplitude":
        F = compute_features(X)[AMPLITUDE]
        return np.column_stack([np.log10(F["peak"]), np.log10(F["rms"]), F["clipped"]])
    if kind == "features":
        return np.log(compute_features(X)[AMPLITUDE + SHAPE].values + 1e-4)
    raise ValueError(f"unknown extra: {kind}")


def prepare(X, train, cfg: CNNConfig):
    """Everything computed from the training rows only: the global scale, and
    the extra inputs for all files standardised with the training mean and spread."""
    A = extra_inputs(X, cfg.extra)
    A = (A - A[train].mean(0)) / (A[train].std(0) + 1e-6)
    return dict(scale=global_scale(X, train), extra=A.astype(np.float32))


class _WithExtra(Dataset):
    """Item i -> (waveform tensor, extra numbers of that file, class id)."""

    def __init__(self, base: WaveformDataset, extra):
        self.base, self.extra = base, extra

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        return x, torch.from_numpy(self.extra[self.base.rows[i]]), y


def train_one(X, y, train, cfg: CNNConfig, seed=0, sessions=None):
    """Train a fresh model on rows `train`. Returns (model, prep).
    `sessions` (one id per row of X) is needed for sampler="session"."""
    torch.manual_seed(seed)
    prep = prepare(X, train, cfg)
    ds = WaveformDataset(X, y, train, crop=cfg.crop, normalize=cfg.normalize, scale=prep["scale"],
                         augment=AUGMENT[cfg.augment], seed=seed)
    if cfg.sampler == "session":
        sampler = session_balanced_sampler(y[train], np.asarray(sessions)[train], seed)
    else:
        sampler = balanced_sampler(y[train], seed)
    dl = DataLoader(_WithExtra(ds, prep["extra"]), batch_size=cfg.batch_size, sampler=sampler)
    model = SmallCNN(width=cfg.width, kernel=cfg.kernel, n_blocks=cfg.n_blocks, dropout=cfg.dropout,
                     n_extra=prep["extra"].shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.epochs * len(dl))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    model.train()
    for _ in range(cfg.epochs):
        for xb, eb, yb in dl:
            opt.zero_grad()
            loss_fn(model(xb, eb), yb).backward()
            opt.step()
            sched.step()
    return model.eval(), prep


@torch.no_grad()
def predict_proba(model, X, rows, cfg: CNNConfig, prep) -> np.ndarray:
    x = preprocess(X[rows], crop=cfg.crop, normalize=cfg.normalize, scale=prep["scale"])
    logits = model(torch.from_numpy(x)[:, None], torch.from_numpy(prep["extra"][rows]))
    return torch.softmax(logits, dim=1).numpy()


def cv_cnn(X, files, cfg: CNNConfig, seeds=range(10), n_folds=N_FOLDS, verbose=True) -> np.ndarray:
    """Out-of-fold class probabilities, one set per seed: (n_seeds, n_files, n_classes)."""
    torch.set_num_threads(THREADS)
    y = files["y"].values
    P = np.zeros((len(seeds), len(files), len(CLASSES)), dtype=np.float32)
    t0 = time.perf_counter()
    for s, seed in enumerate(seeds):
        for k in range(n_folds):
            train, val = fold_split(files, k)
            model, prep = train_one(X, y, train, cfg, seed=seed, sessions=files["session_id"].values)
            P[s, val] = predict_proba(model, X, val, cfg, prep)
    if verbose:
        print(f"{len(seeds)} seeds x {n_folds} folds trained in {time.perf_counter() - t0:.0f} s")
    return P


def cv_cnn_cached(X, files, cfg: CNNConfig, seeds=range(10), cache_dir=None, verbose=True) -> np.ndarray:
    """cv_cnn, but saved to cache_dir/cv_<key>.npy. The key covers the config,
    the seeds, the files, their folds and THREADS, so any change trains again.
    Only non-default settings enter the key, so adding a new setting later
    keeps the old results valid."""
    if cache_dir is None:
        cache_dir = ensure_dirs()["cache"] / "cnn"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = json.dumps([cfg.changes(), list(seeds), list(files["sha256"]), files["fold"].tolist(), THREADS])
    path = cache_dir / f"cv_{hashlib.sha256(key.encode()).hexdigest()[:16]}.npy"
    if path.exists():
        if verbose:
            print("loaded from cache")
        return np.load(path)
    P = cv_cnn(X, files, cfg, seeds=seeds, verbose=verbose)
    np.save(path, P)
    return P
