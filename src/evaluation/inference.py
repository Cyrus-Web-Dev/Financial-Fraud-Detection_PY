"""Inferensi statistik untuk evaluasi model: bootstrap, uji DeLong, uji McNemar, koreksi Holm.

Catatan penting untuk data panel
--------------------------------
Baris dari perusahaan yang sama tidak independen. Karena itu:
* Bootstrap dilakukan pada tingkat PERUSAHAAN (cluster bootstrap): perusahaan diambil ulang beserta
  seluruh barisnya. Bootstrap per baris akan menghasilkan CI yang terlalu sempit.
* Uji DeLong mengasumsikan observasi independen; pada data panel p-value-nya cenderung terlalu kecil.
  Gunakan ``paired_cluster_bootstrap`` sebagai pembanding yang menghormati struktur klaster.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

MetricFn = Callable[[np.ndarray, np.ndarray], float]


# ---------------------------------------------------------------------------
# Bootstrap (cluster)
# ---------------------------------------------------------------------------
def _group_index(groups: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    uniq, inv = np.unique(groups, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    bounds = np.cumsum(np.bincount(inv))[:-1]
    return uniq, np.split(order, bounds)


def _resample_indices(members: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    pick = rng.integers(0, len(members), size=len(members))
    return np.concatenate([members[i] for i in pick])


def cluster_bootstrap(y: np.ndarray, score: np.ndarray, groups: np.ndarray, metric: MetricFn,
                      n_boot: int = 1000, alpha: float = 0.05, seed: int = 42) -> dict[str, float]:
    """CI persentil bootstrap untuk ``metric(y, score)`` dengan pengambilan ulang per grup."""
    y, score, groups = np.asarray(y).astype(int), np.asarray(score, dtype=float), np.asarray(groups)
    if not (len(y) == len(score) == len(groups)):
        raise ValueError("y, score, dan groups harus sama panjang.")
    _, members = _group_index(groups)
    rng = np.random.default_rng(seed)
    point = float(metric(y, score))
    vals = []
    for _ in range(n_boot):
        idx = _resample_indices(members, rng)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        v = metric(yb, score[idx])
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < max(10, n_boot // 10):
        raise ValueError("Terlalu sedikit replikasi bootstrap yang valid.")
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return {"estimasi": point, "ci_bawah": float(lo), "ci_atas": float(hi), "se_bootstrap": float(np.std(vals, ddof=1)),
            "n_replikasi_valid": len(vals)}


def bootstrap_table(y: np.ndarray, score: np.ndarray, groups: np.ndarray, metrics: dict[str, MetricFn],
                    n_boot: int = 1000, seed: int = 42) -> pd.DataFrame:
    rows = [{"metrik": name, **cluster_bootstrap(y, score, groups, fn, n_boot, seed=seed)} for name, fn in metrics.items()]
    return pd.DataFrame(rows)


def paired_cluster_bootstrap(y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray, groups: np.ndarray,
                             metric: MetricFn, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42) -> dict[str, float]:
    """Selisih metrik (A - B) dengan sampel ulang yang SAMA untuk kedua model (berpasangan).

    p-value dua sisi = 2 * min(P(d <= 0), P(d >= 0)) dari distribusi bootstrap (dibatasi minimal 1/n_boot).
    """
    y, a, b, groups = np.asarray(y).astype(int), np.asarray(score_a, float), np.asarray(score_b, float), np.asarray(groups)
    _, members = _group_index(groups)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = _resample_indices(members, rng)
        if y[idx].min() == y[idx].max():
            continue
        d = metric(y[idx], a[idx]) - metric(y[idx], b[idx])
        if np.isfinite(d):
            diffs.append(d)
    diffs = np.asarray(diffs)
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"selisih": float(metric(y, a) - metric(y, b)), "ci_bawah": float(lo), "ci_atas": float(hi),
            "p_value": float(min(1.0, max(p, 1.0 / len(diffs)))), "n_replikasi_valid": int(len(diffs))}


# ---------------------------------------------------------------------------
# DeLong
# ---------------------------------------------------------------------------
def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    z = x[order]
    n = len(x)
    out = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        out[order[i:j]] = 0.5 * (i + j - 1) + 1
        i = j
    return out


def delong_components(y: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """AUC dan matriks kovarians DeLong untuk ``k`` model (Sun & Xu, 2014).

    scores: array (k, n) atau (n,). Mengembalikan (auc[k], cov[k, k]).
    """
    y = np.asarray(y).astype(int)
    S = np.atleast_2d(np.asarray(scores, dtype=float))
    if S.shape[1] != len(y):
        raise ValueError("Skor dan y tidak sama panjang.")
    pos, neg = S[:, y == 1], S[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    if m < 2 or n < 2:
        raise ValueError("DeLong membutuhkan minimal 2 observasi pada tiap kelas.")
    k = S.shape[0]
    tx, ty, tz = np.empty((k, m)), np.empty((k, n)), np.empty((k, m + n))
    for r in range(k):
        tx[r], ty[r] = _midrank(pos[r]), _midrank(neg[r])
        tz[r] = _midrank(np.concatenate([pos[r], neg[r]]))
    auc = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n            # komponen struktural pada kelas positif
    v10 = 1.0 - (tz[:, m:] - ty) / m      # komponen struktural pada kelas negatif
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return auc, cov


def delong_auc_ci(y: np.ndarray, score: np.ndarray, alpha: float = 0.05) -> dict[str, float]:
    auc, cov = delong_components(y, score)
    se = float(np.sqrt(cov[0, 0]))
    z = stats.norm.ppf(1 - alpha / 2)
    return {"auc": float(auc[0]), "se": se, "ci_bawah": float(max(0.0, auc[0] - z * se)),
            "ci_atas": float(min(1.0, auc[0] + z * se))}


def delong_test(y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray, alpha: float = 0.05) -> dict[str, float]:
    """Uji dua sisi H0: AUC_A = AUC_B untuk dua model pada data yang sama (AUC berkorelasi)."""
    auc, cov = delong_components(y, np.vstack([score_a, score_b]))
    diff = float(auc[0] - auc[1])
    var = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])
    if var <= 0:
        return {"auc_a": float(auc[0]), "auc_b": float(auc[1]), "selisih": diff, "se": 0.0, "z": np.nan,
                "p_value": 1.0 if diff == 0 else 0.0, "ci_bawah": diff, "ci_atas": diff}
    se = float(np.sqrt(var))
    z = diff / se
    zc = stats.norm.ppf(1 - alpha / 2)
    return {"auc_a": float(auc[0]), "auc_b": float(auc[1]), "selisih": diff, "se": se, "z": float(z),
            "p_value": float(2 * stats.norm.sf(abs(z))), "ci_bawah": diff - zc * se, "ci_atas": diff + zc * se}


def holm_correction(p_values: Sequence[float]) -> np.ndarray:
    """Koreksi Holm-Bonferroni (kontrol FWER). NaN dipertahankan."""
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if valid.size == 0:
        return out
    order = valid[np.argsort(p[valid])]
    m = len(order)
    adj = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out[order] = np.minimum(adj, 1.0)
    return out


# ---------------------------------------------------------------------------
# McNemar
# ---------------------------------------------------------------------------
def mcnemar_test(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, float | str]:
    """Uji McNemar untuk dua klasifikator pada data yang sama.

    ``b`` = A benar & B salah; ``c`` = A salah & B benar. Uji eksak binomial bila b + c < 25,
    selain itu chi-square dengan koreksi kontinuitas (Edwards).
    """
    y, a, b_ = (np.asarray(v).astype(int) for v in (y, pred_a, pred_b))
    ok_a, ok_b = a == y, b_ == y
    b, c = int((ok_a & ~ok_b).sum()), int((~ok_a & ok_b).sum())
    n_disc = b + c
    if n_disc == 0:
        return {"b": b, "c": c, "statistik": 0.0, "p_value": 1.0, "metode": "tidak ada ketidaksesuaian"}
    if n_disc < 25:
        p = float(stats.binomtest(min(b, c), n_disc, 0.5, alternative="two-sided").pvalue)
        return {"b": b, "c": c, "statistik": float(min(b, c)), "p_value": p, "metode": "eksak binomial"}
    chi2 = (abs(b - c) - 1.0) ** 2 / n_disc
    return {"b": b, "c": c, "statistik": float(chi2), "p_value": float(stats.chi2.sf(chi2, 1)),
            "metode": "chi-square (koreksi kontinuitas)"}
