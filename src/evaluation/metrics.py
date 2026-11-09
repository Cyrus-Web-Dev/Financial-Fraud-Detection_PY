"""Metrik evaluasi klasifikasi biner untuk deteksi fraud.

Tahap ini memuat metrik dasar yang bebas-ambang (ranking) dan metrik pada ambang tertentu.
Analisis lanjutan (kalibrasi, CI bootstrap, uji DeLong/McNemar, ambang berbasis biaya)
ditambahkan pada tahap evaluasi.

Mengapa PR-AUC dan Recall@k?
Pada fraud yang jarang (mis. 6%), accuracy menyesatkan: model yang selalu menjawab "tidak fraud"
sudah 94% akurat. ROC-AUC bisa tampak tinggi karena banyaknya negatif; PR-AUC (average precision)
langsung mencerminkan kualitas pada kelas minoritas, sedangkan Recall@k% menjawab pertanyaan
operasional: "bila auditor hanya memeriksa k% perusahaan teratas, berapa fraud yang tertangkap?"
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def _validate(y: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    if y.shape != score.shape:
        raise ValueError("y dan skor harus sama panjang.")
    if not np.isfinite(score).all():
        raise ValueError("Skor mengandung NaN/inf.")
    return y, score


def recall_precision_at_k(y: np.ndarray, score: np.ndarray, k: float = 0.10) -> tuple[float, float]:
    """(Recall, Precision) bila ``k`` (proporsi) observasi dengan skor tertinggi ditandai fraud."""
    y, score = _validate(y, score)
    if not 0.0 < k <= 1.0:
        raise ValueError("k harus di (0, 1].")
    n_pos = int(y.sum())
    n_top = max(1, int(np.ceil(k * len(y))))
    order = np.argsort(-score, kind="stable")[:n_top]
    tp = int(y[order].sum())
    return (tp / n_pos if n_pos else np.nan), tp / n_top


def ranking_metrics(y: np.ndarray, score: np.ndarray, k: float = 0.10) -> dict[str, float]:
    """Metrik bebas-ambang: ROC-AUC, PR-AUC, Recall@k, Precision@k. NaN bila hanya satu kelas."""
    y, score = _validate(y, score)
    out = {"roc_auc": np.nan, "pr_auc": np.nan, f"recall_at_{int(k * 100)}": np.nan,
           f"precision_at_{int(k * 100)}": np.nan}
    if y.min() == y.max():
        return out
    out["roc_auc"] = float(roc_auc_score(y, score))
    out["pr_auc"] = float(average_precision_score(y, score))
    rec, prec = recall_precision_at_k(y, score, k)
    out[f"recall_at_{int(k * 100)}"], out[f"precision_at_{int(k * 100)}"] = float(rec), float(prec)
    return out


def threshold_metrics(y: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Metrik pada satu ambang: matriks konfusi, precision, recall (sensitivitas), specificity, F1, dst."""
    y, proba = _validate(y, proba)
    pred = (proba >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())

    def div(a: float, b: float) -> float:
        return a / b if b else np.nan

    precision, recall, spec = div(tp, tp + fp), div(tp, tp + fn), div(tn, tn + fp)
    f1 = div(2 * precision * recall, precision + recall) if np.isfinite(precision) and np.isfinite(recall) else np.nan
    return {"threshold": float(threshold), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": precision, "recall": recall, "specificity": spec, "f1": f1,
            "accuracy": div(tp + tn, len(y)),
            "balanced_accuracy": float(np.nanmean([recall, spec])) if np.isfinite(recall) or np.isfinite(spec) else np.nan}


def probability_metrics(y: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """Brier score (makin kecil makin baik) -- hanya bermakna untuk probabilitas sungguhan."""
    y, proba = _validate(y, proba)
    return {"brier": float(brier_score_loss(y, np.clip(proba, 0.0, 1.0)))}
