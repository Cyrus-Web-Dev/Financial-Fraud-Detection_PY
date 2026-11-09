"""Preprocessing fitur yang aman dari data leakage dan kompatibel scikit-learn.

Urutan transformasi (semua parameter dipelajari HANYA dari data latih):

1. inf -> NaN
2. Winsorizing pada kuantil [lower_q, upper_q] (meredam outlier tanpa membuang baris)
3. Imputasi median (median dihitung setelah winsorizing)
4. Standardisasi z-score (opsional): (x - mean) / sd

Kolom dibuang saat ``fit`` bila: seluruhnya kosong, proporsi missing melebihi
``max_missing_share``, atau tidak bervariasi (sd = 0). Alasan pembuangan
tersimpan di ``dropped_`` sehingga dapat didokumentasikan di skripsi.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeaturePreprocessor(BaseEstimator, TransformerMixin):
    """Winsorize -> impute median -> (opsional) standardize.

    Parameters
    ----------
    lower_q, upper_q : kuantil batas winsorizing (default 1% dan 99%).
    scale : jika True lakukan standardisasi z-score.
    max_missing_share : kolom dengan proporsi missing di data latih > nilai ini dibuang.
    """

    def __init__(self, lower_q: float = 0.01, upper_q: float = 0.99, scale: bool = True,
                 max_missing_share: float = 0.40) -> None:
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.scale = scale
        self.max_missing_share = max_missing_share

    # ------------------------------------------------------------------ fit
    def fit(self, X: pd.DataFrame, y=None) -> "FeaturePreprocessor":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeaturePreprocessor membutuhkan pandas DataFrame sebagai input.")
        if not 0.0 <= self.lower_q < self.upper_q <= 1.0:
            raise ValueError("Harus berlaku 0 <= lower_q < upper_q <= 1.")
        if X.shape[0] < 2:
            raise ValueError("Data latih minimal 2 baris.")

        Xn = X.apply(pd.to_numeric, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
        self.input_columns_: list[str] = list(Xn.columns)
        self.dropped_: dict[str, str] = {}
        lower, upper, median, mean, std = {}, {}, {}, {}, {}

        for col in Xn.columns:
            x = Xn[col]
            miss = float(x.isna().mean())
            if miss >= 1.0:
                self.dropped_[col] = "seluruh nilai kosong"
                continue
            if miss > self.max_missing_share:
                self.dropped_[col] = f"missing {miss:.1%} melebihi batas {self.max_missing_share:.0%}"
                continue
            lo, hi = np.nanquantile(x.to_numpy(), [self.lower_q, self.upper_q])
            xw = x.clip(lower=lo, upper=hi)
            med = float(np.nanmedian(xw.to_numpy()))
            xi = xw.fillna(med)
            sd = float(xi.std(ddof=0))
            if not np.isfinite(sd) or sd == 0.0:
                self.dropped_[col] = "tidak bervariasi (sd = 0) pada data latih"
                continue
            lower[col], upper[col], median[col] = float(lo), float(hi), med
            mean[col], std[col] = float(xi.mean()), sd

        self.columns_: list[str] = list(lower.keys())
        self.lower_, self.upper_, self.median_ = lower, upper, median
        self.mean_, self.std_ = mean, std
        self.missing_share_train_ = {c: float(Xn[c].isna().mean()) for c in self.columns_}
        return self

    # ------------------------------------------------------------ transform
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeaturePreprocessor membutuhkan pandas DataFrame sebagai input.")
        absent = [c for c in self.columns_ if c not in X.columns]
        if absent:
            raise ValueError(f"Kolom berikut tidak ada pada data: {', '.join(absent)}")

        out = X[self.columns_].apply(pd.to_numeric, errors="coerce").astype(float)
        out = out.replace([np.inf, -np.inf], np.nan)
        for col in self.columns_:
            x = out[col].clip(lower=self.lower_[col], upper=self.upper_[col]).fillna(self.median_[col])
            if self.scale:
                x = (x - self.mean_[col]) / self.std_[col]
            out[col] = x
        return out

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        self._check_fitted()
        return np.asarray(self.columns_, dtype=object)

    def parameters_table(self) -> pd.DataFrame:
        """Parameter yang dipelajari per fitur (untuk dokumentasi/transparansi)."""
        self._check_fitted()
        return pd.DataFrame({
            "fitur": self.columns_,
            "batas_bawah": [self.lower_[c] for c in self.columns_],
            "batas_atas": [self.upper_[c] for c in self.columns_],
            "median_imputasi": [self.median_[c] for c in self.columns_],
            "mean": [self.mean_[c] for c in self.columns_],
            "sd": [self.std_[c] for c in self.columns_],
            "persen_missing_latih": [self.missing_share_train_[c] * 100 for c in self.columns_],
        })

    def _check_fitted(self) -> None:
        if not hasattr(self, "columns_"):
            raise RuntimeError("FeaturePreprocessor belum di-fit. Panggil fit() terlebih dahulu.")
