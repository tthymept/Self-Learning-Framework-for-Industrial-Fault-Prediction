"""Scoring models with the session-based folds (shared by baseline and CNN).

    pred = cv_predict(make_model, F, files)   # out-of-fold predictions
    per_fold, overall = score(files["y"], pred, files["fold"])
"""
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score

from .data import CLASSES, fold_split
from .split import N_FOLDS


def cv_predict(make_model, F, files, n_folds=N_FOLDS) -> np.ndarray:
    """Train a fresh model per fold; each file is predicted by the model that did not see it."""
    F = np.asarray(F)
    y = files["y"].values
    pred = np.empty_like(y)
    for k in range(n_folds):
        train, val = fold_split(files, k)
        pred[val] = make_model().fit(F[train], y[train]).predict(F[val])
    return pred


def metrics(y, pred) -> dict:
    labels = range(len(CLASSES))
    out = {
        "macro_f1": f1_score(y, pred, labels=labels, average="macro", zero_division=0),
        "balanced_acc": balanced_accuracy_score(y, pred),
        "accuracy": float(np.mean(np.asarray(y) == np.asarray(pred))),
    }
    rec = recall_score(y, pred, labels=labels, average=None, zero_division=0)
    out.update({f"recall_{c}": r for c, r in zip(CLASSES, rec)})
    return out


def balanced_log_loss(y, proba, eps=1e-6) -> float:
    """-log(probability given to the true class), averaged within each class,
    then over classes. Lower is better; guessing 1/4 for everything = log 4 = 1.39.
    Unlike macro-F1 it also moves when a file is right but barely, so it is a
    smoother signal for comparing models on few files."""
    y = np.asarray(y)
    nll = -np.log(np.clip(np.asarray(proba)[np.arange(len(y)), y], eps, 1))
    return float(np.mean([nll[y == c].mean() for c in np.unique(y)]))


def session_balanced_accuracy(y, pred, sessions) -> float:
    """Share correct within each recording session, averaged over the sessions of
    a class, then over classes. Every session counts once, so a 62-file session
    that goes right or wrong together weighs the same as a 1-file session."""
    df = pd.DataFrame({"y": np.asarray(y), "ok": np.asarray(y) == np.asarray(pred), "s": np.asarray(sessions)})
    per_session = df.groupby("s").agg(y=("y", "first"), ok=("ok", "mean"))
    return float(per_session.groupby("y")["ok"].mean().mean())


def score(y, pred, folds):
    """(per-fold metrics table, metrics over all out-of-fold predictions)."""
    y, pred, folds = map(np.asarray, (y, pred, folds))
    per_fold = pd.DataFrame(
        {f"fold {k}": metrics(y[folds == k], pred[folds == k]) for k in np.unique(folds)}
    ).T
    return per_fold, metrics(y, pred)


def confusion(y, pred) -> np.ndarray:
    return confusion_matrix(y, pred, labels=range(len(CLASSES)))
