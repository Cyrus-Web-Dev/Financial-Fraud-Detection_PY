"""Nilai Shapley eksak (interventional) untuk penjelasan model.

Definisi
--------
Untuk model f, instance x, dan sampel latar (background) B, nilai koalisi S adalah

    v(S) = rata-rata_b f(x_S, b_{bukan S})

yaitu fitur dalam S diambil dari x dan sisanya dari baris latar b. Nilai Shapley fitur j:

    phi_j = sum_{S tanpa j}  |S|! (d-|S|-1)! / d!  *  [ v(S + j) - v(S) ]

Sifat yang dijamin (dan diuji):
* Efisiensi : sum_j phi_j = f(x) - E_b f(b)   (``expected_value``)
* Dummy     : fitur yang tidak dipakai model mendapat 0
* Simetri   : fitur yang berperan identik mendapat nilai sama
* Aditivitas: phi(f + g) = phi(f) + phi(g)
* Model linear f = w.x + c  ->  phi_j = w_j (x_j - rata-rata_b x_j)

Ini adalah definisi yang sama dengan ``shap.TreeExplainer(model, data=B, feature_perturbation=
"interventional")`` dan ``shap.explainers.Exact``. Implementasi ini MENGENUMERASI seluruh 2^d koalisi,
sehingga eksak tetapi biaya naik eksponensial terhadap d. Dibatasi d <= 14 (untuk d lebih besar gunakan
pustaka ``shap`` dengan estimator pohon/aproksimasi). Pada proyek ini d = 10 (1.024 koalisi).

Catatan: karena pencampuran dilakukan per kolom utuh, atribusi pada fitur yang sudah ditransformasi
per-kolom (winsorize, imputasi, standardisasi) identik dengan atribusi pada fitur aslinya.
"""
from __future__ import annotations

from math import factorial
from typing import Callable, Sequence

import numpy as np

MAX_FEATURES = 14


class ExactShapleyExplainer:
    """Penjelas Shapley eksak model-agnostik.

    Parameters
    ----------
    predict_fn : fungsi array (n, d) -> vektor (n,) skor skalar (mis. P(fraud) atau log-odds).
    background : array (m, d) sampel latar (mis. sebagian data latih).
    feature_names : nama d fitur.
    """

    def __init__(self, predict_fn: Callable[[np.ndarray], np.ndarray], background: np.ndarray,
                 feature_names: Sequence[str]) -> None:
        B = np.asarray(background, dtype=float)
        if B.ndim != 2 or B.shape[0] < 1:
            raise ValueError("background harus array 2 dimensi tidak kosong.")
        d = B.shape[1]
        if d != len(feature_names):
            raise ValueError("Jumlah fitur_names tidak sama dengan kolom background.")
        if d > MAX_FEATURES:
            raise ValueError(f"Shapley eksak dibatasi d <= {MAX_FEATURES} (diberikan {d}). Kurangi fitur atau gunakan pustaka shap.")
        if not np.isfinite(B).all():
            raise ValueError("background mengandung NaN/inf.")
        self.predict_fn, self.background, self.feature_names, self.d = predict_fn, B, list(feature_names), d

        n_coal = 1 << d
        ids = np.arange(n_coal)
        self._masks = ((ids[:, None] >> np.arange(d)) & 1).astype(bool)            # (2^d, d)
        w = np.array([factorial(s) * factorial(d - s - 1) / factorial(d) for s in range(d)])
        self._idx0, self._idx1, self._w = [], [], []
        for j in range(d):
            without = ids[(ids >> j) & 1 == 0]
            self._idx0.append(without)
            self._idx1.append(without | (1 << j))
            self._w.append(w[self._masks[without].sum(axis=1)])
        self.expected_value = float(np.mean(self._call(B)))

    def _call(self, A: np.ndarray) -> np.ndarray:
        out = np.asarray(self.predict_fn(A), dtype=float).reshape(-1)
        if out.shape[0] != A.shape[0] or not np.isfinite(out).all():
            raise ValueError("predict_fn harus mengembalikan vektor hingga sepanjang jumlah baris.")
        return out

    def coalition_values(self, x: np.ndarray) -> np.ndarray:
        """v(S) untuk seluruh 2^d koalisi (indeks bitmask: bit j = fitur j termasuk S)."""
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.shape[0] != self.d:
            raise ValueError("Panjang x tidak sama dengan jumlah fitur.")
        m = self.background.shape[0]
        mixed = np.where(self._masks[:, None, :], x[None, None, :], self.background[None, :, :])   # (2^d, m, d)
        return self._call(mixed.reshape(-1, self.d)).reshape(-1, m).mean(axis=1)

    def shap_values_one(self, x: np.ndarray) -> np.ndarray:
        v = self.coalition_values(x)
        return np.array([np.sum(self._w[j] * (v[self._idx1[j]] - v[self._idx0[j]])) for j in range(self.d)])

    def shap_values(self, X: np.ndarray) -> np.ndarray:
        """Matriks atribusi (n, d). Untuk tiap baris: sum(phi) + expected_value = f(x)."""
        A = np.asarray(X, dtype=float)
        if A.ndim != 2 or A.shape[1] != self.d:
            raise ValueError(f"X harus berbentuk (n, {self.d}).")
        return np.vstack([self.shap_values_one(row) for row in A])


def linear_shap(coef: np.ndarray, X: np.ndarray, background: np.ndarray) -> tuple[np.ndarray, float]:
    """Nilai Shapley eksak untuk model linear (skala log-odds): phi_j = w_j (x_j - rata2 latar_j).

    Mengembalikan (phi[n, d], intersep-bebas offset); ``expected_value`` = intersep + w . rata2(latar),
    dihitung oleh pemanggil karena bergantung pada intersep model.
    """
    w = np.asarray(coef, dtype=float).reshape(-1)
    A, B = np.asarray(X, dtype=float), np.asarray(background, dtype=float)
    if A.shape[1] != len(w) or B.shape[1] != len(w):
        raise ValueError("Dimensi koefisien, X, dan latar tidak sesuai.")
    mean_b = B.mean(axis=0)
    return (A - mean_b) * w, float(w @ mean_b)


def model_scorer(estimator, output: str = "proba") -> Callable[[np.ndarray], np.ndarray]:
    """Bungkus classifier (setelah preprocessing) menjadi ``predict_fn`` skalar.

    output = "proba"  -> P(fraud)
    output = "logit"  -> log-odds, ln(p / (1 - p)) dengan p dipotong ke [1e-6, 1 - 1e-6]
    """
    if output not in {"proba", "logit"}:
        raise ValueError("output harus 'proba' atau 'logit'.")

    def fn(A: np.ndarray) -> np.ndarray:
        p = np.asarray(estimator.predict_proba(A))[:, 1]
        if output == "proba":
            return p
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    return fn
