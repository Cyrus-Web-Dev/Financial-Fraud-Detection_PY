"""Uji diagnostik dan asumsi regresi logistik.

* Hosmer-Lemeshow : kecocokan model (goodness-of-fit) berbasis kelompok risiko.
* Box-Tidwell     : asumsi linearitas logit untuk setiap prediktor kontinu.
* Pengaruh        : leverage, residual Pearson dan deviance, Cook's Distance.
* Multikolinearitas: lihat ``src.features.selection.compute_vif``.

Rumus (model dengan matriks desain A, p = probabilitas fit, w = p(1-p), W = diag(w)):
  leverage      h_i = w_i a_i' (A'WA)^-1 a_i                (sum h_i = jumlah parameter k)
  Pearson       r_i = (y_i - p_i) / sqrt(w_i)
  deviance      d_i = sign(y_i - p_i) sqrt(-2 [y_i ln p_i + (1-y_i) ln(1-p_i)])   (sum d_i^2 = -2 ln L)
  Cook          D_i = r_i^2 h_i / (k (1 - h_i)^2)            (aproksimasi satu langkah, Pregibon 1981)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.models.logit import CONST, LogitInference


# ---------------------------------------------------------------------------
# Hosmer-Lemeshow
# ---------------------------------------------------------------------------
def hosmer_lemeshow(y: np.ndarray, p: np.ndarray, groups: int = 10) -> dict:
    """Uji Hosmer-Lemeshow.

    Observasi diurutkan menurut p dan dibagi ke ``groups`` kelompok berukuran (hampir) sama.
        C = sum_g (O_g - E_g)^2 / (E_g (1 - E_g / n_g)),  C ~ chi2(g - 2) di bawah H0.
    H0: model cocok dengan data. p-value KECIL berarti model TIDAK cocok.
    Uji ini bergantung pada pengelompokan dan lemah pada sampel kecil; laporkan bersama kurva kalibrasi.
    """
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    if y.shape != p.shape or groups < 3:
        raise ValueError("y dan p harus sama panjang dan groups >= 3.")
    if len(y) < groups * 5:
        raise ValueError("Sampel terlalu kecil untuk jumlah kelompok yang diminta.")
    order = np.argsort(p, kind="stable")
    rows, stat = [], 0.0
    for g, idx in enumerate(np.array_split(order, groups), start=1):
        n_g, obs, exp = len(idx), float(y[idx].sum()), float(p[idx].sum())
        pi_bar = exp / n_g
        denom = n_g * pi_bar * (1.0 - pi_bar)
        stat += (obs - exp) ** 2 / denom if denom > 0 else 0.0
        rows.append({"kelompok": g, "n": n_g, "fraud_teramati": obs, "fraud_diharapkan": exp,
                     "rata_prob": pi_bar})
    dof = groups - 2
    return {"statistik": float(stat), "dof": dof, "p_value": float(stats.chi2.sf(stat, dof)),
            "tabel": pd.DataFrame(rows)}


# ---------------------------------------------------------------------------
# Box-Tidwell
# ---------------------------------------------------------------------------
def box_tidwell(X: pd.DataFrame, y: np.ndarray, features: list[str] | None = None, alpha: float = 0.05
                ) -> pd.DataFrame:
    """Uji Box-Tidwell untuk linearitas logit.

    Untuk tiap prediktor kontinu x_j ditambahkan suku x_j * ln(x_j) ke model; bila koefisien suku itu
    signifikan, hubungan logit-x_j TIDAK linear (perlu transformasi: log, kuadrat, dsb.).
    Karena x_j harus positif, nilai digeser agar minimum = 1 (``digeser`` = True) -- ini mengubah
    skala tetapi tidak mengubah kesimpulan tentang linearitas terhadap transformasi pangkat.
    Ambang signifikansi dikoreksi Bonferroni (alpha / jumlah prediktor yang diuji).
    """
    feats = list(features) if features is not None else list(X.columns)
    y = np.asarray(y).astype(int)
    base = X[list(X.columns)].reset_index(drop=True).astype(float)
    rows = []
    thr = alpha / max(len(feats), 1)
    for f in feats:
        x = base[f].to_numpy()
        shifted = bool(x.min() <= 0)
        xp = x - x.min() + 1.0 if shifted else x
        design = base.copy()
        design[f] = xp
        design[f"{f}__xlnx"] = xp * np.log(xp)
        m = LogitInference().fit(design, y)
        term = f"{f}__xlnx"
        z = float(m.params_[term] / m.bse_[term])
        rows.append({"fitur": f, "koef_xlnx": float(m.params_[term]), "z": z, "p_value": float(2 * stats.norm.sf(abs(z))),
                     "ambang_bonferroni": thr, "linear": bool(2 * stats.norm.sf(abs(z)) >= thr), "digeser": shifted})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pengaruh dan residual
# ---------------------------------------------------------------------------
def influence_diagnostics(model: LogitInference, X: pd.DataFrame | np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Leverage, residual Pearson/deviance, dan Cook's Distance untuk setiap observasi."""
    yv = np.asarray(y, dtype=float)
    A = model._design(X)
    k = A.shape[1]
    p = np.clip(model.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
    w = p * (1 - p)
    info = A.T @ (A * w[:, None])
    inv_info = np.linalg.inv(info)
    h = w * np.einsum("ij,jk,ik->i", A, inv_info, A)
    pearson = (yv - p) / np.sqrt(w)
    dev = np.sign(yv - p) * np.sqrt(-2.0 * (yv * np.log(p) + (1 - yv) * np.log(1 - p)))
    cook = pearson ** 2 * h / (k * (1 - h) ** 2)
    return pd.DataFrame({"prob": p, "leverage": h, "resid_pearson": pearson, "resid_deviance": dev,
                         "cooks_d": cook}, index=X.index if isinstance(X, pd.DataFrame) else None)


def influence_summary(diag: pd.DataFrame, n_params: int) -> dict:
    """Ringkasan observasi berpengaruh dengan aturan praktis yang lazim.

    * Cook's D > 4/n  (tanda untuk diperiksa; > 1 sangat berpengaruh)
    * leverage > 2k/n
    * |residual deviance| > 2 (sekitar 5% dari observasi diharapkan melewatinya bila model cocok)
    """
    n = len(diag)
    return {"n": n, "cooks_d_gt_4n": int((diag["cooks_d"] > 4.0 / n).sum()), "cooks_d_gt_1": int((diag["cooks_d"] > 1.0).sum()),
            "cooks_d_maks": float(diag["cooks_d"].max()),
            "leverage_gt_2k_per_n": int((diag["leverage"] > 2.0 * n_params / n).sum()),
            "leverage_maks": float(diag["leverage"].max()),
            "resid_deviance_abs_gt_2": int((diag["resid_deviance"].abs() > 2).sum()),
            "resid_deviance_abs_gt_3": int((diag["resid_deviance"].abs() > 3).sum()),
            "persen_resid_deviance_gt_2": float((diag["resid_deviance"].abs() > 2).mean() * 100)}


def residual_normality_note() -> str:
    """Catatan metodologi untuk skripsi."""
    return ("Residual regresi logistik untuk data biner tidak berdistribusi normal; uji normalitas residual "
            "(Shapiro-Wilk, dsb.) TIDAK relevan. Gunakan residual deviance/Pearson untuk mencari observasi "
            "yang tidak cocok dan Hosmer-Lemeshow serta kurva kalibrasi untuk kecocokan model.")


__all__ = ["hosmer_lemeshow", "box_tidwell", "influence_diagnostics", "influence_summary", "CONST"]
