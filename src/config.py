"""Konfigurasi global proyek Financial Fraud Detection.

Semua path dan konstanta penting dikumpulkan di sini agar tidak ada
"magic value" yang tersebar di banyak file (reproducible research).
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Path
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
SAMPLE_DIR: Path = DATA_DIR / "sample"
MODELS_DIR: Path = BASE_DIR / "models"
REPORTS_DIR: Path = BASE_DIR / "reports"

UPLOAD_DIR: Path = DATA_DIR / "uploads"
HISTORY_DB_FILE: Path = DATA_DIR / "history.db"

SAMPLE_CLEAN_FILE: Path = SAMPLE_DIR / "sample_fraud_data.csv"
SAMPLE_DIRTY_FILE: Path = SAMPLE_DIR / "sample_fraud_data_dirty.csv"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Parameter data sintetis (default)
# ---------------------------------------------------------------------------
SYNTH_N_FIRMS: int = 600
SYNTH_YEARS: tuple[int, ...] = (2018, 2019, 2020, 2021, 2022, 2023)
SYNTH_FRAUD_PREVALENCE: float = 0.06  # target proporsi fraud (~6%)

# ---------------------------------------------------------------------------
# Keamanan upload (dipakai di tahap API)
# ---------------------------------------------------------------------------
MAX_UPLOAD_MB: int = 20

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
MAX_PREDICT_ROWS: int = 5000         # batas baris per permintaan prediksi (penjelasan per baris dibatasi terpisah)
EXPLAIN_MAX_ROWS: int = 50           # penjelasan per baris hanya untuk <= jumlah ini (biaya komputasi)
MAX_STORED_DATASETS: int = 20        # dataset unggahan yang disimpan di server (yang terlama dihapus)
CORS_ORIGINS: tuple[str, ...] = ("http://localhost:5173", "http://127.0.0.1:5173")
