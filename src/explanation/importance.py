"""Ringkasan kepentingan fitur: SHAP global, dependence, permutation importance, kesepakatan antar model."""
from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats


def mean_abs_shap(phi: np.ndarray, names: Sequence[str]) -> pd.DataFrame:
    """Kepentingan global = rata-rata |phi|; ``rata_bertanda`` menunjukkan arah rata-rata efeknya."""
    phi = np.asarray(phi, dtype=float)
    imp = np.abs(phi).mean(axis=0)
    out = pd.DataFrame({"fitur": list(names), "mean_abs_shap": imp, "rata_bertanda": phi.mean(axis=0),
                        "porsi": imp / imp.sum() if imp.sum() > 0 else np.nan})
    out = out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    out["peringkat"] = np.arange(1, len(out) + 1)
    return out


def beeswarm_data(phi: np.ndarray, X: pd.DataFrame, max_points_per_feature: int = 300, seed: int = 42) -> pd.DataFrame:
    """Data format panjang untuk summary plot: nilai fitur (persentil 0-1) vs nilai SHAP."""
    rng = np.random.default_rng(seed)
    n = len(X)
    take = np.sort(rng.choice(n, size=min(n, max_points_per_feature), replace=False))
    frames = []
    for j, name in enumerate(X.columns):
        x = X[name].to_numpy(dtype=float)
        pct = stats.rankdata(x) / n
        frames.append(pd.DataFrame({"fitur": name, "baris": take, "nilai_fitur": x[take], "persentil_nilai": pct[take],
                                    "shap": phi[take, j]}))
    return pd.concat(frames, ignore_index=True)


def dependence_data(phi: np.ndarray, X: pd.DataFrame, feature: str, color: str | None = None,
                    n_bins: int = 5) -> dict:
    """Data dependence plot untuk ``feature``; warna otomatis = fitur yang paling menjelaskan sisa variasi SHAP.

    Skor interaksi untuk fitur kandidat k: bagi x_j ke ``n_bins`` kuantil, hitung sisa phi_j (dikurangi rata-rata
    per kuantil), lalu rata-ratakan |korelasi Spearman| antara sisa itu dan x_k DI DALAM tiap kuantil (tertimbang
    ukuran kuantil). Pengukuran di dalam kuantil diperlukan agar interaksi yang membalik tanda (efek k positif
    saat x_j > 0 dan negatif saat x_j < 0) tidak saling menghapus pada korelasi global.
    """
    cols = list(X.columns)
    j = cols.index(feature)
    x = X[feature].to_numpy(dtype=float)
    pj = np.asarray(phi)[:, j]
    if color is None:
        bins = pd.qcut(pd.Series(x).rank(method="first"), q=min(n_bins, len(x)), labels=False).to_numpy()
        resid = pj - pd.Series(pj).groupby(bins).transform("mean").to_numpy()
        best, best_score = None, -1.0
        for k, c in enumerate(cols):
            if k == j:
                continue
            xk = X[c].to_numpy(dtype=float)
            total, weight = 0.0, 0
            for b in np.unique(bins):
                m = bins == b
                if m.sum() < 5 or np.ptp(xk[m]) == 0 or np.ptp(resid[m]) == 0:
                    continue
                r = stats.spearmanr(resid[m], xk[m]).statistic
                if np.isfinite(r):
                    total += abs(r) * m.sum(); weight += m.sum()
            score = total / weight if weight else 0.0
            if score > best_score:
                best, best_score = c, score
        color = best
    out = {"fitur": feature, "warna_fitur": color, "x": x.tolist(), "shap": pj.tolist()}
    if color is not None:
        out["nilai_warna"] = X[color].to_numpy(dtype=float).tolist()
    return out


def permutation_importance(predict_fn: Callable[[pd.DataFrame], np.ndarray], X: pd.DataFrame, y: np.ndarray,
                           metric: Callable[[np.ndarray, np.ndarray], float], n_repeats: int = 10,
                           seed: int = 42) -> pd.DataFrame:
    """Penurunan metrik saat satu kolom diacak (model-agnostik; dihitung pada data uji).

    Kepentingan = metrik dasar - metrik setelah pengacakan (positif = fitur berguna). Pengacakan memutus
    hubungan fitur dengan label; fitur yang berkorelasi kuat saling menutupi sehingga kepentingannya
    bisa terlihat kecil -- tafsirkan bersama SHAP dan VIF.
    """
    y = np.asarray(y).astype(int)
    base = float(metric(y, predict_fn(X)))
    rng = np.random.default_rng(seed)
    rows = []
    for col in X.columns:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].to_numpy())
            drops.append(base - float(metric(y, predict_fn(Xp))))
        rows.append({"fitur": col, "penurunan_mean": float(np.mean(drops)),
                     "penurunan_sd": float(np.std(drops, ddof=1)) if n_repeats > 1 else np.nan})
    out = pd.DataFrame(rows).sort_values("penurunan_mean", ascending=False).reset_index(drop=True)
    out["metrik_dasar"] = base
    return out


def rank_agreement(importances: Mapping[str, pd.Series], top_k: int = 3) -> dict:
    """Kesepakatan peringkat kepentingan antar model/metode: Spearman dan irisan top-k.

    ``importances``: nama metode -> Series (indeks = fitur, nilai = kepentingan, lebih besar = lebih penting).
    """
    names = list(importances)
    feats = sorted(set.intersection(*(set(s.index) for s in importances.values())))
    mat = pd.DataFrame({k: importances[k].reindex(feats).astype(float) for k in names})
    rho = pd.DataFrame(np.nan, index=names, columns=names)
    overlap = pd.DataFrame(np.nan, index=names, columns=names)
    top = {k: set(mat[k].nlargest(top_k).index) for k in names}
    for a in names:
        for b in names:
            rho.loc[a, b] = 1.0 if a == b else stats.spearmanr(mat[a], mat[b]).statistic
            overlap.loc[a, b] = len(top[a] & top[b]) / top_k
    return {"spearman": rho, f"irisan_top{top_k}": overlap, "top_fitur": {k: sorted(v) for k, v in top.items()}}
