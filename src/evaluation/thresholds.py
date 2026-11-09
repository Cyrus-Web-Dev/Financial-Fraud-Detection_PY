"""Pemilihan ambang keputusan dan analisis kurva keputusan.

Pada deteksi fraud, kerugian false negative (fraud lolos) jauh lebih besar daripada false positive
(audit yang sia-sia). Ambang 0,5 hampir tidak pernah tepat. Tersedia tiga aturan:

* ``youden``  : memaksimalkan J = sensitivitas + spesifisitas - 1.
* ``max_f1``  : memaksimalkan F1 kelas fraud.
* ``cost``    : meminimalkan total biaya C_FN * FN + C_FP * FP.
  Untuk probabilitas TERKALIBRASI, ambang optimal teoretis adalah t* = C_FP / (C_FP + C_FN)
  (``bayes_threshold``); pada skor yang tidak terkalibrasi, ambang dicari secara empiris.

Penting: ambang dipilih pada data LATIH (prediksi out-of-fold), lalu diterapkan ke data uji.
Memilih ambang pada data uji adalah kebocoran informasi.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.metrics import threshold_metrics


def _validate(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y, s = np.asarray(y).astype(int), np.asarray(s, dtype=float)
    if y.shape != s.shape or y.ndim != 1:
        raise ValueError("y dan skor harus vektor sama panjang.")
    if not np.isfinite(s).all():
        raise ValueError("Skor mengandung NaN/inf.")
    if y.min() == y.max():
        raise ValueError("Diperlukan kedua kelas.")
    return y, s


def threshold_scan(y: np.ndarray, s: np.ndarray, max_points: int = 400) -> pd.DataFrame:
    """Matriks konfusi dan metrik untuk setiap ambang kandidat (nilai skor unik, maks ``max_points``).

    Aturan prediksi: fraud bila skor >= ambang.
    """
    y, s = _validate(y, s)
    uniq = np.unique(s)
    if len(uniq) > max_points:
        uniq = np.unique(np.quantile(s, np.linspace(0, 1, max_points)))
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order], s[order]
    tp_cum, fp_cum = np.cumsum(ys), np.cumsum(1 - ys)
    # jumlah observasi dengan skor >= t  ->  indeks terakhir yang skor-nya >= t
    cut = np.searchsorted(-ss, -uniq, side="right")
    tp, fp = tp_cum[cut - 1], fp_cum[cut - 1]
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    fn, tn = n_pos - tp, n_neg - fp
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
        recall = tp / n_pos
        f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0.0)
    return pd.DataFrame({"ambang": uniq, "tp": tp, "fp": fp, "tn": tn, "fn": fn, "recall": recall,
                         "specificity": tn / n_neg, "precision": precision, "f1": f1})


def youden_threshold(y: np.ndarray, s: np.ndarray) -> dict[str, float]:
    t = threshold_scan(y, s, max_points=100000)
    j = t["recall"] + t["specificity"] - 1.0
    i = int(j.idxmax())
    return {"ambang": float(t.loc[i, "ambang"]), "youden_j": float(j[i])}


def max_f1_threshold(y: np.ndarray, s: np.ndarray) -> dict[str, float]:
    t = threshold_scan(y, s, max_points=100000)
    i = int(t["f1"].idxmax())
    return {"ambang": float(t.loc[i, "ambang"]), "f1": float(t.loc[i, "f1"])}


def bayes_threshold(c_fn: float, c_fp: float) -> float:
    """Ambang optimal teoretis untuk probabilitas terkalibrasi: C_FP / (C_FP + C_FN)."""
    if c_fn <= 0 or c_fp <= 0:
        raise ValueError("Biaya harus positif.")
    return c_fp / (c_fp + c_fn)


def cost_optimal_threshold(y: np.ndarray, s: np.ndarray, c_fn: float = 10.0, c_fp: float = 1.0) -> dict[str, float]:
    """Ambang yang meminimalkan total biaya empiris. Bila seri, pilih ambang TERTINGGI (paling sedikit ditandai)."""
    if c_fn <= 0 or c_fp <= 0:
        raise ValueError("Biaya harus positif.")
    y, s = _validate(y, s)
    t = threshold_scan(y, s, max_points=100000)
    cost = c_fn * t["fn"] + c_fp * t["fp"]
    best = cost[cost == cost.min()].index.max()
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    return {"ambang": float(t.loc[best, "ambang"]), "biaya_total": float(cost[best]),
            "biaya_per_observasi": float(cost[best] / len(y)),
            "biaya_tanpa_model": float(c_fn * n_pos),          # tidak menandai siapa pun
            "biaya_tandai_semua": float(c_fp * n_neg), "c_fn": float(c_fn), "c_fp": float(c_fp)}


def apply_threshold(y: np.ndarray, s: np.ndarray, threshold: float, c_fn: float = 10.0, c_fp: float = 1.0
                    ) -> dict[str, float]:
    """Metrik dan biaya pada ambang tetap (mis. ambang dari data latih diterapkan ke data uji)."""
    y = np.asarray(y).astype(int)
    m = threshold_metrics(y, np.asarray(s, dtype=float), threshold)
    m["biaya_total"] = float(c_fn * m["fn"] + c_fp * m["fp"])
    m["biaya_per_observasi"] = m["biaya_total"] / len(y)
    m["biaya_tanpa_model"] = float(c_fn * y.sum())
    m["biaya_tanpa_model_per_observasi"] = m["biaya_tanpa_model"] / len(y)
    m["penghematan_biaya_persen"] = float(100.0 * (1.0 - m["biaya_total"] / m["biaya_tanpa_model"])) if m["biaya_tanpa_model"] > 0 else float("nan")
    m["persen_ditandai"] = float((m["tp"] + m["fp"]) / len(y))
    return m


def decision_curve(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray | None = None) -> pd.DataFrame:
    """Decision curve analysis (Vickers & Elkin, 2006).

    Net benefit(t) = TP/n - FP/n * t/(1-t). Dibandingkan dengan strategi "tandai semua" dan
    "tidak tandai siapa pun" (NB = 0). Model berguna pada ambang t bila net benefit-nya melampaui keduanya.
    ``p`` sebaiknya probabilitas terkalibrasi agar ambang t bermakna sebagai kesediaan menerima risiko.
    """
    y, p = _validate(y, p)
    th = np.linspace(0.01, 0.40, 40) if thresholds is None else np.asarray(thresholds, dtype=float)
    if ((th <= 0) | (th >= 1)).any():
        raise ValueError("Ambang harus di antara 0 dan 1 (eksklusif).")
    n, prev = len(y), float(y.mean())
    rows = []
    for t in th:
        pred = p >= t
        tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
        w = t / (1.0 - t)
        rows.append({"ambang": float(t), "net_benefit_model": tp / n - fp / n * w,
                     "net_benefit_tandai_semua": prev - (1.0 - prev) * w, "net_benefit_tidak_ada": 0.0})
    return pd.DataFrame(rows)


def downsample_curve(x: np.ndarray, y: np.ndarray, max_points: int = 200) -> tuple[list[float], list[float]]:
    """Kurangi jumlah titik kurva (ROC/PR) untuk dikirim ke frontend."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) <= max_points:
        return x.tolist(), y.tolist()
    idx = np.unique(np.linspace(0, len(x) - 1, max_points).astype(int))
    return x[idx].tolist(), y[idx].tolist()
