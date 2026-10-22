"""Stage 3 (models): hand-made features for the baseline model (one row per waveform).

AMPLITUDE features depend on the signal size in volts (and therefore on the
recorder's gain and trigger level). SHAPE features do not: they stay the same
if a waveform is scaled up or down.
"""
import numpy as np
import pandas as pd

from ..dataset.loading import FS, TRIGGER

BANDS_MHZ = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 40), (40, 62.5)]
AMPLITUDE = ["peak", "rms", "clipped"]
SHAPE = (
    ["crest", "kurtosis", "centroid_MHz"]
    + [f"band_{a:g}_{b:g}MHz" for a, b in BANDS_MHZ]
    + ["energy_near_trigger", "energy_late"]
)


def us(n):
    return int(round(n * FS / 1e6))


def compute_features(X: np.ndarray) -> pd.DataFrame:
    D = X - np.median(X, axis=1, keepdims=True)
    E = D ** 2
    rms = np.sqrt(E.mean(axis=1))
    peak = np.abs(D).max(axis=1)

    F = np.abs(np.fft.rfft(D, axis=1)) ** 2
    f = np.fft.rfftfreq(X.shape[1], 1 / FS) / 1e6

    out = {
        "peak": peak,
        "rms": rms,
        "clipped": (np.abs(X) >= 0.999).mean(axis=1),
        "crest": peak / rms,
        "kurtosis": (D ** 4).mean(axis=1) / E.mean(axis=1) ** 2,
        "centroid_MHz": (F * f).sum(axis=1) / F.sum(axis=1),
    }
    for a, b in BANDS_MHZ:
        m = (f >= a) & (f < b)
        out[f"band_{a:g}_{b:g}MHz"] = F[:, m].sum(axis=1) / F.sum(axis=1)
    total = E.sum(axis=1)
    out["energy_near_trigger"] = E[:, TRIGGER - us(1):TRIGGER + us(2)].sum(axis=1) / total
    out["energy_late"] = E[:, TRIGGER + us(10):].sum(axis=1) / total
    return pd.DataFrame(out)
