"""Perakit ringkasan EDA lengkap (satu panggilan untuk halaman EDA di frontend)."""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.data.quality import detect_outliers
from src.data.schema import CURRENT_COLUMNS, TARGET_COLUMN
from src.evaluation.benford import benford_by_column, benford_by_group
from src.evaluation.eda import (
    class_distribution,
    compare_groups,
    correlation_matrix,
    descriptive_statistics,
    high_correlation_pairs,
    normality_tests,
)
from src.utils import to_jsonable


def build_eda_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Ringkasan EDA untuk data yang sudah divalidasi (tipe numerik sudah benar)."""
    has_target = TARGET_COLUMN in df.columns and pd.to_numeric(df[TARGET_COLUMN], errors="coerce").notna().any()
    cols = [c for c in CURRENT_COLUMNS if c in df.columns]

    corr = correlation_matrix(df, cols, method="spearman")
    summary: dict[str, Any] = {
        "deskriptif": descriptive_statistics(df, cols),
        "normalitas": normality_tests(df, cols),
        "korelasi": {"kolom": list(corr.columns), "matriks": corr.to_numpy()},
        "korelasi_tinggi": high_correlation_pairs(corr, threshold=0.8),
        "outlier": detect_outliers(df),
        "benford_per_kolom": benford_by_column(df, cols),
        "benford_per_kelompok": benford_by_group(df, cols),
    }
    if has_target:
        summary["distribusi_kelas"] = class_distribution(df)
        summary["distribusi_per_tahun"] = class_distribution(df, by="tahun")
        summary["distribusi_per_sektor"] = class_distribution(df, by="sektor")
        summary["perbandingan_kelompok"] = compare_groups(df, cols)
    return to_jsonable(summary)
