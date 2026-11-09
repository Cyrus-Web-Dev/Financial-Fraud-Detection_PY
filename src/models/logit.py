"""Regresi logistik dengan inferensi statistik lengkap (IRLS / Newton-Raphson).

Menyediakan apa yang dibutuhkan skripsi Statistika dari model parametrik:
koefisien, standard error, statistik Wald z, p-value, interval kepercayaan,
odds ratio beserta CI-nya, log-likelihood, uji rasio likelihood (LR) terhadap
model nol, pseudo-R2 McFadden, AIC, dan BIC.

Mengapa ditulis sendiri (bukan Statsmodels)?
Hasilnya diverifikasi terhadap scikit-learn (estimasi tanpa penalti), Hessian numerik
(standard error), dan solusi analitik (Firth model intersep saja). Bila Statsmodels
terpasang, ``tests/test_logit.py`` juga membandingkan secara langsung.

Estimasi
--------
* MLE: maksimalkan log L(b) = sum[y log p + (1-y) log(1-p)], p = expit(Xb), dengan
  IRLS:  b <- b + (X'WX)^-1 X'(y - p),  W = diag(p(1-p)), plus step-halving agar
  log-likelihood tidak turun.
* Firth (``firth=True``): penalized likelihood dengan penalti Jeffreys 0.5 log|X'WX|.
  Skor dimodifikasi menjadi X'(y - p + h(0.5 - p)), h = diagonal matriks topi.
  Menghasilkan estimasi hingga pada kasus separasi sempurna dan mengurangi bias
  kecil-sampel pada kejadian langka (Firth, 1993) -- relevan untuk fraud.

Standard error dihitung dari invers informasi Fisher X'WX di titik estimasi
(untuk Firth: pendekatan Wald pada estimasi terpenalti). Log-likelihood yang
dilaporkan adalah log-likelihood biasa di titik estimasi.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit

CONST = "const"


def _loglik(y: np.ndarray, eta: np.ndarray) -> float:
    """Log-likelihood Bernoulli yang stabil secara numerik."""
    return float(np.sum(y * -np.logaddexp(0.0, -eta) + (1.0 - y) * -np.logaddexp(0.0, eta)))


class LogitInference:
    """Regresi logistik dengan inferensi.

    Parameters
    ----------
    firth : gunakan penalized likelihood Firth.
    add_constant : tambahkan intersep otomatis (kolom bernama ``const``).
    max_iter, tol : batas iterasi dan toleransi (max |langkah|).
    alpha : tingkat signifikansi CI (0.05 -> CI 95%).
    """

    def __init__(self, firth: bool = False, add_constant: bool = True, max_iter: int = 100,
                 tol: float = 1e-8, alpha: float = 0.05) -> None:
        self.firth = firth
        self.add_constant = add_constant
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha

    # ------------------------------------------------------------------ fit
    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray | pd.Series) -> "LogitInference":
        names = list(X.columns) if isinstance(X, pd.DataFrame) else [f"x{i}" for i in range(np.asarray(X).shape[1])]
        A = np.asarray(X, dtype=float)
        yv = np.asarray(y, dtype=float)
        if A.ndim != 2 or A.shape[0] != yv.shape[0]:
            raise ValueError("X harus 2 dimensi dan sama panjang dengan y.")
        if not (np.isfinite(A).all() and np.isfinite(yv).all()):
            raise ValueError("X dan y tidak boleh mengandung NaN/inf.")
        if not np.isin(yv, [0.0, 1.0]).all() or yv.min() == yv.max():
            raise ValueError("y harus biner 0/1 dengan kedua kelas hadir.")
        if self.add_constant:
            if CONST in names:
                raise ValueError(f"Nama kolom '{CONST}' dipakai untuk intersep.")
            A = np.column_stack([np.ones(len(A)), A])
            names = [CONST] + names
        n, k = A.shape
        if n <= k:
            raise ValueError("Jumlah observasi harus lebih besar dari jumlah parameter.")

        beta = np.zeros(k)
        if self.add_constant:
            p0 = yv.mean()
            beta[0] = np.log(p0 / (1 - p0))
        converged = False
        singular = False
        n_iter = 0

        def penalized_ll(b: np.ndarray) -> float:
            eta_ = A @ b
            ll_ = _loglik(yv, eta_)
            if self.firth:
                w_ = np.clip(expit(eta_) * (1 - expit(eta_)), 1e-12, None)
                sign, logdet = np.linalg.slogdet(A.T @ (A * w_[:, None]))
                ll_ += 0.5 * logdet if sign > 0 else -np.inf
            return ll_

        current = penalized_ll(beta)
        for n_iter in range(1, self.max_iter + 1):
            eta = A @ beta
            p = expit(eta)
            w = np.clip(p * (1 - p), 1e-12, None)
            info = A.T @ (A * w[:, None])
            if self.firth:
                try:
                    inv_info = np.linalg.inv(info)
                except np.linalg.LinAlgError:
                    inv_info = np.linalg.pinv(info)
                h = w * np.einsum("ij,jk,ik->i", A, inv_info, A)
                score = A.T @ (yv - p + h * (0.5 - p))
            else:
                score = A.T @ (yv - p)
            try:
                step = np.linalg.solve(info, score)
            except np.linalg.LinAlgError:
                singular = True
                step = np.linalg.pinv(info) @ score
            # step-halving: pastikan (penalized) log-likelihood tidak turun
            scale = 1.0
            for _ in range(30):
                cand = beta + scale * step
                val = penalized_ll(cand)
                if np.isfinite(val) and val >= current - 1e-10:
                    break
                scale /= 2.0
            else:
                cand, val = beta, current
            max_step = float(np.max(np.abs(cand - beta)))
            beta, current = cand, val
            if max_step < self.tol:
                converged = True
                break

        eta = A @ beta
        p = expit(eta)
        w = np.clip(p * (1 - p), 1e-12, None)
        info = A.T @ (A * w[:, None])
        try:
            cov = np.linalg.inv(info)
        except np.linalg.LinAlgError:
            singular = True
            cov = np.linalg.pinv(info)

        ll = _loglik(yv, eta)
        n1 = yv.sum()
        ll_null = float(n1 * np.log(n1 / n) + (n - n1) * np.log((n - n1) / n))
        k_exog = k - (1 if self.add_constant else 0)

        self.feature_names_ = names
        self.params_ = pd.Series(beta, index=names)
        self.cov_ = pd.DataFrame(cov, index=names, columns=names)
        self.bse_ = pd.Series(np.sqrt(np.clip(np.diag(cov), 0, None)), index=names)
        self.llf_ = ll
        self.llnull_ = ll_null
        self.nobs_ = n
        self.n_params_ = k
        self.n_iter_ = n_iter
        self.converged_ = converged and not singular
        self.singular_ = singular
        self.lr_stat_ = max(0.0, 2.0 * (ll - ll_null))
        self.lr_df_ = k_exog
        self.lr_pvalue_ = float(stats.chi2.sf(self.lr_stat_, k_exog)) if k_exog > 0 else np.nan
        self.pseudo_r2_ = 1.0 - ll / ll_null
        self.aic_ = -2.0 * ll + 2.0 * k
        self.bic_ = -2.0 * ll + k * np.log(n)
        fitted = p
        # separasi: koefisien meledak, atau probabilitas fit sangat dekat 0/1 untuk kelas yang tepat
        perfect = np.mean(((fitted > 1 - 1e-8) & (yv == 1)) | ((fitted < 1e-8) & (yv == 0)))
        self.separation_suspected_ = bool(((np.abs(beta) > 15).any() or perfect > 0.01) and not self.firth)
        return self

    # -------------------------------------------------------------- predict
    def _design(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        self._check()
        if isinstance(X, pd.DataFrame):
            cols = [c for c in self.feature_names_ if c != CONST]
            missing = [c for c in cols if c not in X.columns]
            if missing:
                raise ValueError(f"Kolom tidak ada: {', '.join(missing)}")
            A = X[cols].to_numpy(dtype=float)
        else:
            A = np.asarray(X, dtype=float)
        if self.add_constant:
            A = np.column_stack([np.ones(len(A)), A])
        if A.shape[1] != len(self.params_):
            raise ValueError("Jumlah kolom X tidak sesuai dengan model.")
        return A

    def decision_function(self, X) -> np.ndarray:
        return self._design(X) @ self.params_.to_numpy()

    def predict_proba(self, X) -> np.ndarray:
        p1 = expit(self.decision_function(X))
        return np.column_stack([1 - p1, p1])

    # -------------------------------------------------------------- summary
    def summary(self) -> pd.DataFrame:
        """Tabel koefisien: b, SE, z, p, CI, odds ratio, dan CI odds ratio."""
        self._check()
        zcrit = stats.norm.ppf(1 - self.alpha / 2)
        b, se = self.params_, self.bse_
        z = b / se
        with np.errstate(over="ignore"):      # koefisien ekstrem (mis. kolinearitas berat) -> odds ratio = inf
            out = pd.DataFrame({
                "koefisien": b, "std_error": se, "z": z,
                "p_value": 2 * stats.norm.sf(np.abs(z)),
                "ci_bawah": b - zcrit * se, "ci_atas": b + zcrit * se,
                "odds_ratio": np.exp(b),
                "or_ci_bawah": np.exp(b - zcrit * se), "or_ci_atas": np.exp(b + zcrit * se),
            })
        out.index.name = "variabel"
        return out

    def model_fit(self) -> dict[str, float | int | bool]:
        """Statistik kecocokan model keseluruhan."""
        self._check()
        return {
            "n_obs": int(self.nobs_), "n_parameter": int(self.n_params_),
            "log_likelihood": float(self.llf_), "log_likelihood_null": float(self.llnull_),
            "lr_chi2": float(self.lr_stat_), "lr_df": int(self.lr_df_), "lr_p_value": float(self.lr_pvalue_),
            "pseudo_r2_mcfadden": float(self.pseudo_r2_), "aic": float(self.aic_), "bic": float(self.bic_),
            "iterasi": int(self.n_iter_), "konvergen": bool(self.converged_),
            "separasi_dicurigai": bool(self.separation_suspected_), "firth": bool(self.firth),
        }

    def _check(self) -> None:
        if not hasattr(self, "params_"):
            raise RuntimeError("Model belum di-fit.")


def add_unit_odds_ratios(summary: pd.DataFrame, sd: dict[str, float] | pd.Series) -> pd.DataFrame:
    """Tambahkan odds ratio per SATU SATUAN ASLI fitur dari model yang dilatih pada fitur terstandardisasi.

    Pada model standardisasi, ``odds_ratio`` bermakna per kenaikan 1 simpangan baku (SD).
    Odds ratio per satuan asli = exp(b / SD), dengan CI dikonversi dengan cara yang sama.
    """
    out = summary.copy()
    sd = pd.Series(sd, dtype=float)
    idx = [i for i in out.index if i in sd.index]
    out["sd_fitur"] = sd.reindex(out.index)
    for col, src in (("or_per_satuan", "koefisien"), ("or_satuan_ci_bawah", "ci_bawah"), ("or_satuan_ci_atas", "ci_atas")):
        out[col] = np.nan
        out.loc[idx, col] = np.exp(out.loc[idx, src] / sd[idx])
    return out


def likelihood_ratio_test(full: LogitInference, reduced: LogitInference) -> dict[str, float]:
    """Uji LR antar dua model bersarang (``reduced`` bagian dari ``full``) pada data yang sama."""
    if full.nobs_ != reduced.nobs_:
        raise ValueError("Kedua model harus dipasang pada data yang sama.")
    df = full.n_params_ - reduced.n_params_
    if df <= 0:
        raise ValueError("Model penuh harus punya lebih banyak parameter daripada model reduksi.")
    stat = max(0.0, 2.0 * (full.llf_ - reduced.llf_))
    return {"lr_chi2": float(stat), "df": int(df), "p_value": float(stats.chi2.sf(stat, df))}


def feature_names_without_const(model: LogitInference) -> Sequence[str]:
    return [c for c in model.feature_names_ if c != CONST]
