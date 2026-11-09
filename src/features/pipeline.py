"""Perakit tabel fitur: gabungan identitas, label, fitur model, dan skor benchmark."""
from __future__ import annotations

import pandas as pd

from src.data.schema import IDENTIFIER_COLUMNS, TARGET_COLUMN
from src.features.ratios import FEATURE_NAMES, compute_benchmark_scores, compute_ratios

BENCHMARK_COLUMNS: list[str] = ["m_score", "m_flag", "altman_z", "altman_z_zone", "altman_z2", "altman_z2_zone"]


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """Kolom: identitas + (label jika ada) + ``FEATURE_NAMES`` + skor benchmark.

    ``df`` harus sudah divalidasi (tipe numerik benar). Indeks baris dipertahankan.
    """
    parts = [df[IDENTIFIER_COLUMNS]]
    if TARGET_COLUMN in df.columns:
        parts.append(df[[TARGET_COLUMN]])
    parts += [compute_ratios(df), compute_benchmark_scores(df)[BENCHMARK_COLUMNS]]
    return pd.concat(parts, axis=1)


__all__ = ["build_feature_table", "BENCHMARK_COLUMNS", "FEATURE_NAMES"]
