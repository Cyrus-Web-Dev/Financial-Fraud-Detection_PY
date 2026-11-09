"""Membuat file data sampel (bersih dan "kotor") untuk demo.

Jalankan dari root proyek:
    python -m scripts.generate_sample_data
"""
from __future__ import annotations

from src.config import SAMPLE_CLEAN_FILE, SAMPLE_DIR, SAMPLE_DIRTY_FILE
from src.data.schema import TARGET_COLUMN
from src.data.synthetic import generate_financial_panel, inject_data_issues


def main() -> None:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    clean = generate_financial_panel()
    clean.to_csv(SAMPLE_CLEAN_FILE, index=False, float_format="%.0f")

    dirty = inject_data_issues(clean)
    dirty.to_csv(SAMPLE_DIRTY_FILE, index=False, float_format="%.0f")

    n = len(clean)
    n_fraud = int(clean[TARGET_COLUMN].sum())
    print(f"Data bersih : {SAMPLE_CLEAN_FILE}")
    print(f"Data kotor  : {SAMPLE_DIRTY_FILE}")
    print(f"Jumlah baris: {n} | perusahaan: {clean['company_id'].nunique()}")
    print(f"Fraud       : {n_fraud} baris ({n_fraud / n:.2%})")


if __name__ == "__main__":
    main()
