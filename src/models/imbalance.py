"""Penanganan ketidakseimbangan kelas (class imbalance).

Strategi (``strategy``)
-----------------------
* ``none``         : tanpa penanganan (baseline pembanding).
* ``class_weight`` : bobot kelas 'balanced' pada fungsi loss; data tidak diubah.
* ``random_over``  : duplikasi acak kelas minoritas sampai rasio target.
* ``random_under`` : pembuangan acak kelas mayoritas sampai rasio target.
* ``smote``        : Synthetic Minority Over-sampling Technique (Chawla dkk., 2002);
  sampel sintetis dibuat dengan interpolasi linear antara sebuah sampel
  minoritas dan salah satu dari k tetangga terdekatnya.

``sampling_ratio`` = rasio akhir n_minoritas / n_mayoritas (1.0 = seimbang penuh).

Pencegahan data leakage
-----------------------
``ResamplingClassifier`` melakukan resampling DI DALAM ``fit``. Saat dipakai dalam
validasi silang, hanya data latih tiap fold yang di-resample; data validasi tidak
pernah disentuh. Meresampling seluruh data SEBELUM membagi fold adalah kesalahan
klasik: salinan/interpolasi dari sampel yang sama masuk ke latih dan validasi,
sehingga metrik tampak jauh lebih baik dari yang sebenarnya.

Catatan: setelah resampling atau pembobotan, probabilitas keluaran model bias ke atas
(prior kelas minoritas "diperbesar"). Peringkat (ROC-AUC, PR-AUC) tetap valid, tetapi
probabilitas perlu dikalibrasi ulang (lihat tahap evaluasi/kalibrasi).
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.neighbors import NearestNeighbors

STRATEGIES: tuple[str, ...] = ("none", "class_weight", "random_over", "random_under", "smote")


def _check_binary(y: np.ndarray) -> tuple[int, int]:
    classes = np.unique(y)
    if not np.array_equal(classes, [0, 1]):
        raise ValueError("Resampling mengharuskan label biner 0/1 dengan kedua kelas hadir.")
    return int((y == 1).sum()), int((y == 0).sum())


def random_oversample(X: np.ndarray, y: np.ndarray, ratio: float = 1.0, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    n_min, n_maj = _check_binary(y)
    target = int(round(ratio * n_maj))
    n_new = max(0, target - n_min)
    if n_new == 0:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(np.flatnonzero(y == 1), size=n_new, replace=True)
    return np.vstack([X, X[idx]]), np.concatenate([y, y[idx]])


def random_undersample(X: np.ndarray, y: np.ndarray, ratio: float = 1.0, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    n_min, n_maj = _check_binary(y)
    keep_maj = min(n_maj, int(round(n_min / ratio)))
    if keep_maj >= n_maj:
        return X, y
    rng = np.random.default_rng(seed)
    maj_keep = rng.choice(np.flatnonzero(y == 0), size=keep_maj, replace=False)
    keep = np.sort(np.concatenate([np.flatnonzero(y == 1), maj_keep]))
    return X[keep], y[keep]


def smote(X: np.ndarray, y: np.ndarray, ratio: float = 1.0, k_neighbors: int = 5, seed: int = 42
          ) -> tuple[np.ndarray, np.ndarray]:
    """SMOTE untuk fitur numerik kontinu. Jika minoritas < 2 sampel, jatuh ke random oversampling."""
    n_min, n_maj = _check_binary(y)
    n_new = max(0, int(round(ratio * n_maj)) - n_min)
    if n_new == 0:
        return X, y
    if n_min < 2:
        return random_oversample(X, y, ratio, seed)
    rng = np.random.default_rng(seed)
    Xm = X[y == 1]
    k = min(k_neighbors, n_min - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Xm)
    neighbors = nn.kneighbors(Xm, return_distance=False)[:, 1:]  # buang diri sendiri

    base = rng.integers(0, n_min, size=n_new)
    pick = neighbors[base, rng.integers(0, k, size=n_new)]
    lam = rng.random((n_new, 1))
    synthetic = Xm[base] + lam * (Xm[pick] - Xm[base])
    return np.vstack([X, synthetic]), np.concatenate([y, np.ones(n_new, dtype=y.dtype)])


def class_counts(y: np.ndarray) -> dict[str, int]:
    y = np.asarray(y)
    return {"non_fraud": int((y == 0).sum()), "fraud": int((y == 1).sum())}


def resample(X: np.ndarray, y: np.ndarray, strategy: str, ratio: float = 1.0, k_neighbors: int = 5,
             seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Terapkan strategi resampling (``none`` dan ``class_weight`` tidak mengubah data)."""
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy harus salah satu dari {STRATEGIES}.")
    if not 0.0 < ratio <= 1.0:
        raise ValueError("sampling_ratio harus di (0, 1].")
    if strategy in ("none", "class_weight"):
        return X, y
    if strategy == "random_over":
        return random_oversample(X, y, ratio, seed)
    if strategy == "random_under":
        return random_undersample(X, y, ratio, seed)
    return smote(X, y, ratio, k_neighbors, seed)


class ResamplingClassifier(BaseEstimator, ClassifierMixin):
    """Pembungkus estimator: resampling/pembobotan dilakukan saat ``fit`` saja.

    Parameters
    ----------
    estimator : classifier scikit-learn dasar.
    strategy : salah satu dari ``STRATEGIES``.
    sampling_ratio : rasio minoritas/mayoritas akhir untuk strategi resampling.
    k_neighbors : jumlah tetangga SMOTE.
    random_state : seed resampling.
    """

    def __init__(self, estimator: Any = None, strategy: str = "none", sampling_ratio: float = 1.0,
                 k_neighbors: int = 5, random_state: int = 42) -> None:
        self.estimator = estimator
        self.strategy = strategy
        self.sampling_ratio = sampling_ratio
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit(self, X, y):
        if self.estimator is None:
            raise ValueError("estimator wajib diisi.")
        if self.strategy not in STRATEGIES:
            raise ValueError(f"strategy harus salah satu dari {STRATEGIES}.")
        Xa = np.asarray(X, dtype=float)
        ya = np.asarray(y).astype(int)
        est = clone(self.estimator)
        if self.strategy == "class_weight":
            if "class_weight" not in est.get_params():
                raise ValueError(f"{type(est).__name__} tidak mendukung class_weight.")
            est.set_params(class_weight="balanced")
        Xr, yr = resample(Xa, ya, self.strategy, self.sampling_ratio, self.k_neighbors, self.random_state)
        est.fit(Xr, yr)
        self.estimator_ = est
        self.classes_ = np.array([0, 1])
        self.n_train_original_ = class_counts(ya)
        self.n_train_resampled_ = class_counts(yr)
        return self

    def _check(self) -> None:
        if not hasattr(self, "estimator_"):
            raise RuntimeError("Model belum di-fit.")

    def predict_proba(self, X):
        self._check()
        return self.estimator_.predict_proba(np.asarray(X, dtype=float))

    def decision_function(self, X):
        self._check()
        Xa = np.asarray(X, dtype=float)
        if hasattr(self.estimator_, "decision_function"):
            return self.estimator_.decision_function(Xa)
        return self.estimator_.predict_proba(Xa)[:, 1]

    def predict(self, X):
        self._check()
        return self.estimator_.predict(np.asarray(X, dtype=float))
