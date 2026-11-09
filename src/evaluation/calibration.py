"""Analisis dan perbaikan kalibrasi probabilitas.

Model dengan AUC tinggi belum tentu menghasilkan probabilitas yang dapat dipercaya. Terlebih lagi
setelah pembobotan kelas atau resampling, probabilitas keluaran model bias ke atas.

Ukuran yang dilaporkan
----------------------
* Tabel/kurva reliabilitas (bin kuantil): rata-rata prediksi vs frekuensi fraud teramati.
* ECE (Expected Calibration Error): rata-rata tertimbang |teramati - prediksi| antar bin.
* Slope dan intersep kalibrasi (Cox, 1958): regresi logistik y ~ logit(p). Kalibrasi sempurna:
  slope = 1, intersep = 0. Slope < 1 berarti prediksi terlalu ekstrem (overfitting);
  intersep (calibration-in-the-large, slope dikunci 1) > 0 berarti risiko diremehkan.
* Brier score.

Perbaikan: Platt scaling (regresi logistik pada logit skor) dan regresi isotonik. Kalibrator HARUS
dilatih pada data yang TIDAK dipakai untuk evaluasi akhir (di sini: prediksi out-of-fold data latih)
dan dievaluasi pada data uji.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression

from src.evaluation.eda import wilson_interval
from src.models.logit import LogitInference

EPS = 1e-6


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)


def _check(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError("y dan p harus vektor sama panjang.")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("p harus berupa probabilitas dalam [0, 1] tanpa NaN.")
    return y, p


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Tabel reliabilitas dengan bin berisi jumlah observasi (hampir) sama (kuantil)."""
    y, p = _check(y, p)
    order = np.argsort(p, kind="stable")
    rows = []
    for b, idx in enumerate(np.array_split(order, min(n_bins, len(y))), start=1):
        if len(idx) == 0:
            continue
        k, n = int(y[idx].sum()), len(idx)
        lo, hi = wilson_interval(k, n)
        rows.append({"bin": b, "n": n, "rata_prediksi": float(p[idx].mean()), "frekuensi_teramati": k / n,
                     "ci95_bawah": lo, "ci95_atas": hi})
    return pd.DataFrame(rows)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    t = calibration_table(y, p, n_bins)
    return float((t["n"] * (t["frekuensi_teramati"] - t["rata_prediksi"]).abs()).sum() / t["n"].sum())


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray, alpha: float = 0.05) -> dict[str, float]:
    """Slope dan intersep kalibrasi + calibration-in-the-large.

    * ``slope`` dan ``intersep``: koefisien regresi logistik y ~ a + b * logit(p).
    * ``citl``: intersep a pada model y ~ offset(logit(p)) + a (slope dikunci 1), diselesaikan dengan
      persamaan skor sum(y - expit(z + a)) = 0.
    """
    y, p = _check(y, p)
    if y.min() == y.max():
        raise ValueError("Kalibrasi memerlukan kedua kelas.")
    z = logit(_clip(p))
    m = LogitInference(alpha=alpha).fit(z.reshape(-1, 1), y)
    s = m.summary()
    citl = optimize.brentq(lambda a: float(np.sum(y - expit(z + a))), -20.0, 20.0)
    return {"slope": float(s.loc["x0", "koefisien"]), "slope_ci_bawah": float(s.loc["x0", "ci_bawah"]),
            "slope_ci_atas": float(s.loc["x0", "ci_atas"]), "intersep": float(s.loc["const", "koefisien"]),
            "citl": float(citl), "rata_prediksi": float(p.mean()), "frekuensi_teramati": float(y.mean())}


def calibration_report(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    """Ringkasan: Brier, ECE, slope, intersep, CITL."""
    y, p = _check(y, p)
    brier = float(np.mean((p - y) ** 2))
    base = float(np.mean((y.mean() - y) ** 2))            # Brier model yang selalu menjawab prevalensi
    out = {"brier": brier, "brier_skill_score": 1.0 - brier / base if base > 0 else np.nan,
           "ece": expected_calibration_error(y, p, n_bins)}
    out.update(calibration_slope_intercept(y, p))
    return out


class PlattCalibrator:
    """Platt scaling: P(y=1|s) = expit(a + b * f(s)), f = logit(clip(s)) bila ``input_kind='probability'``."""

    def __init__(self, input_kind: str = "probability") -> None:
        if input_kind not in {"probability", "score"}:
            raise ValueError("input_kind harus 'probability' atau 'score'.")
        self.input_kind = input_kind

    def _f(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        return logit(_clip(s)) if self.input_kind == "probability" else s

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        self.model_ = LogitInference().fit(self._f(scores).reshape(-1, 1), np.asarray(y).astype(int))
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("Kalibrator belum di-fit.")
        return self.model_.predict_proba(self._f(scores).reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    """Kalibrasi isotonik (monoton non-menurun, nonparametrik). Rentan overfit pada sampel kecil."""

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        self.model_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
            np.asarray(scores, dtype=float), np.asarray(y).astype(float))
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("Kalibrator belum di-fit.")
        return np.asarray(self.model_.predict(np.asarray(scores, dtype=float)), dtype=float)
