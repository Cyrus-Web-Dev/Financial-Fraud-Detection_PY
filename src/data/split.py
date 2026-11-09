"""Pembagian data latih/uji yang sadar struktur panel (perusahaan x tahun).

Mengapa tidak memakai split acak biasa?
Satu perusahaan muncul pada beberapa tahun dengan karakteristik yang sangat
mirip. Split acak per baris menaruh perusahaan yang sama di data latih dan
uji, sehingga model "mengenali" perusahaan, bukan mempelajari pola fraud.
Akibatnya metrik uji terlalu optimistis (data leakage).

Tersedia dua strategi:
* ``group_stratified_split`` : seluruh baris satu perusahaan masuk ke satu sisi saja,
  dengan proporsi fraud dijaga seimbang (StratifiedGroupKFold).
* ``temporal_split``         : melatih pada tahun lampau, menguji pada tahun terbaru
  (menyerupai penggunaan nyata: memprediksi laporan yang belum pernah dilihat).

Semua fungsi mengembalikan indeks POSISI (bukan label indeks) sehingga aman
dipakai dengan ``DataFrame.iloc``.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.data.schema import TARGET_COLUMN


def group_stratified_cv(y: np.ndarray | pd.Series, groups: np.ndarray | pd.Series, n_splits: int = 5,
                        seed: int = 42) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Lipatan CV stratified yang tidak memecah satu grup (perusahaan) ke dua sisi."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    if n_splits < 2:
        raise ValueError("n_splits minimal 2.")
    for cls in np.unique(y):
        n_groups = len(np.unique(groups[y == cls]))
        if n_groups < n_splits:
            raise ValueError(
                f"Kelas {cls} hanya tersebar pada {n_groups} perusahaan; kurangi n_splits (< {n_groups + 1}).")
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    yield from cv.split(np.zeros(len(y)), y, groups)


def group_stratified_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42,
                           target: str = TARGET_COLUMN, group_col: str = "company_id"
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Bagi data menjadi latih/uji per perusahaan dengan proporsi fraud yang dijaga."""
    if not 0.05 <= test_size <= 0.5:
        raise ValueError("test_size harus di antara 0.05 dan 0.5.")
    n_splits = max(2, int(round(1.0 / test_size)))
    train_idx, test_idx = next(group_stratified_cv(df[target].to_numpy(), df[group_col].to_numpy(),
                                                   n_splits=n_splits, seed=seed))
    return np.sort(train_idx), np.sort(test_idx)


def temporal_split(df: pd.DataFrame, n_test_years: int = 1, year_col: str = "tahun"
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Latih pada tahun-tahun awal, uji pada ``n_test_years`` tahun terakhir."""
    years = np.sort(df[year_col].unique())
    if n_test_years < 1 or n_test_years >= len(years):
        raise ValueError(f"n_test_years harus antara 1 dan {len(years) - 1} (tersedia {len(years)} tahun).")
    test_years = years[-n_test_years:]
    is_test = df[year_col].isin(test_years).to_numpy()
    return np.flatnonzero(~is_test), np.flatnonzero(is_test)


def split_summary(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray,
                  target: str = TARGET_COLUMN, group_col: str = "company_id") -> dict:
    """Ringkasan pembagian: ukuran, proporsi fraud, dan jumlah perusahaan yang beririsan."""
    tr, te = df.iloc[train_idx], df.iloc[test_idx]
    shared = set(tr[group_col]) & set(te[group_col])
    return {
        "n_latih": int(len(tr)), "n_uji": int(len(te)),
        "fraud_latih": int(tr[target].sum()), "fraud_uji": int(te[target].sum()),
        "proporsi_fraud_latih": float(tr[target].mean()), "proporsi_fraud_uji": float(te[target].mean()),
        "perusahaan_latih": int(tr[group_col].nunique()), "perusahaan_uji": int(te[group_col].nunique()),
        "perusahaan_beririsan": int(len(shared)),
        "tahun_latih": [int(tr["tahun"].min()), int(tr["tahun"].max())] if "tahun" in df.columns else None,
        "tahun_uji": [int(te["tahun"].min()), int(te["tahun"].max())] if "tahun" in df.columns else None,
    }
