"""Deteksi outlier yang tidak bergantung pada ukuran perusahaan.

Nilai mentah laporan keuangan sangat miring dan ukuran perusahaan berbeda
ratusan kali lipat, sehingga outlier tidak bisa dideteksi langsung dari nilai
mentah. Setiap pos diubah menjadi rasio terhadap total aset (tahun yang sama
atau sebelumnya), lalu dinilai dengan *modified z-score* berbasis MAD
(Iglewicz & Hoaglin, 1993), yang tahan terhadap outlier itu sendiri:

    M_i = 0.6745 * (r_i - median(r)) / MAD(r)

Sebuah sel ditandai outlier hanya jika |M| melewati ambang pada KEDUA versi
rasio (terhadap total aset t dan t-1). Dengan begitu, satu sel total aset
yang rusak tidak menandai seluruh baris sebagai outlier.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.schema import CURRENT_COLUMNS, NUMERIC_COLUMNS

DEFAULT_OUTLIER_THRESHOLD: float = 8.0


def modified_z_score(x: pd.Series) -> pd.Series:
    """Modified z-score berbasis MAD. Mengembalikan NaN jika MAD = 0."""
    values = pd.to_numeric(x, errors="coerce").astype(float)
    med = np.nanmedian(values.to_numpy()) if values.notna().any() else np.nan
    mad = np.nanmedian(np.abs(values.to_numpy() - med)) if values.notna().any() else np.nan
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(np.nan, index=x.index)
    return 0.6745 * (values - med) / mad


def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.where(den > 0)  # penyebut non-positif -> NaN
    return num / den


def outlier_mask(df: pd.DataFrame, threshold: float = DEFAULT_OUTLIER_THRESHOLD) -> pd.DataFrame:
    """Matriks boolean (baris x kolom numerik) berisi True untuk sel outlier.

    Kolom yang tidak ada di ``df`` dilewati. Perlu ``total_assets`` dan
    ``total_assets_prev``; jika tidak ada, mengembalikan matriks False.
    """
    cols = [c for c in NUMERIC_COLUMNS if c in df.columns]
    mask = pd.DataFrame(False, index=df.index, columns=cols)
    if "total_assets" not in df.columns or "total_assets_prev" not in df.columns:
        return mask

    ta = pd.to_numeric(df["total_assets"], errors="coerce").astype(float)
    ta_prev = pd.to_numeric(df["total_assets_prev"], errors="coerce").astype(float)

    for col in cols:
        x = pd.to_numeric(df[col], errors="coerce").astype(float)
        if col == "total_assets":
            z1 = modified_z_score(_ratio(x, ta_prev))
            z2 = z1
        elif col == "total_assets_prev":
            z1 = modified_z_score(_ratio(x, ta))
            z2 = z1
        else:
            z1 = modified_z_score(_ratio(x, ta))
            z2 = modified_z_score(_ratio(x, ta_prev))
        flagged = (z1.abs() > threshold) & (z2.abs() > threshold)
        mask[col] = flagged.fillna(False).to_numpy()
    return mask


def detect_outliers(df: pd.DataFrame, threshold: float = DEFAULT_OUTLIER_THRESHOLD) -> pd.DataFrame:
    """Ringkasan jumlah outlier per kolom (hanya kolom dengan outlier > 0)."""
    mask = outlier_mask(df, threshold)
    n = len(df)
    counts = mask.sum(axis=0)
    out = pd.DataFrame({
        "kolom": counts.index,
        "jumlah_outlier": counts.to_numpy(dtype=int),
        "persen": (counts.to_numpy(dtype=float) / max(n, 1) * 100.0),
    })
    out = out[out["jumlah_outlier"] > 0].sort_values("jumlah_outlier", ascending=False)
    return out.reset_index(drop=True)


__all__ = ["modified_z_score", "outlier_mask", "detect_outliers", "DEFAULT_OUTLIER_THRESHOLD",
           "CURRENT_COLUMNS"]
