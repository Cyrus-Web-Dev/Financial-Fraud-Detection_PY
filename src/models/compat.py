"""Kompatibilitas lintas versi scikit-learn.

Di scikit-learn >= 1.8 parameter ``penalty`` pada LogisticRegression
dideprekasi (dihapus di 1.10) dan digantikan ``l1_ratio``. Di versi < 1.8,
``l1_ratio`` DIABAIKAN kecuali penalty='elasticnet', sehingga memakai
``l1_ratio=1.0`` di versi lama akan diam-diam menghasilkan model L2.
Helper ini memilih API yang benar sesuai versi terpasang.
"""
from __future__ import annotations

import re

import sklearn
from sklearn.linear_model import LogisticRegression


def sklearn_version() -> tuple[int, int]:
    m = re.match(r"(\d+)\.(\d+)", sklearn.__version__)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def make_logreg(penalty: str = "l2", C: float = 1.0, class_weight=None, solver: str | None = None,
                max_iter: int = 2000, random_state: int | None = 42, **kwargs) -> LogisticRegression:
    """Buat LogisticRegression dengan penalti 'l1' atau 'l2' yang benar di semua versi.

    L1 memakai solver 'liblinear' secara default (stabil untuk data kecil-menengah).
    """
    if penalty not in {"l1", "l2"}:
        raise ValueError("penalty harus 'l1' atau 'l2'.")
    if solver is None:
        solver = "liblinear" if penalty == "l1" else "lbfgs"
    common = dict(C=C, class_weight=class_weight, solver=solver, max_iter=max_iter,
                  random_state=random_state, **kwargs)
    if sklearn_version() >= (1, 8):
        return LogisticRegression(l1_ratio=1.0 if penalty == "l1" else 0.0, **common)
    return LogisticRegression(penalty=penalty, **common)
