"""Tahap 3: validasi -> fitur -> split per perusahaan -> seleksi fitur (hanya data latih).

    python -m scripts.run_feature_engineering                 # data sampel bersih
    python -m scripts.run_feature_engineering path/data.csv   # data sendiri

Keluaran di data/processed/: features.csv, feature_selection.json, feature_catalog.json, split.json
"""
from __future__ import annotations

import json
import sys

from src.config import PROCESSED_DIR, RANDOM_SEED, SAMPLE_CLEAN_FILE
from src.data.loader import load_dataset
from src.data.split import group_stratified_split, split_summary
from src.data.schema import TARGET_COLUMN
from src.data.validation import validate_dataset
from src.features.pipeline import build_feature_table
from src.features.ratios import FEATURE_NAMES, catalog_as_records
from src.features.selection import run_feature_selection


def main(path: str | None = None, n_boot: int = 100) -> None:
    df, report = validate_dataset(load_dataset(path or SAMPLE_CLEAN_FILE), require_target=True)
    if not report.is_valid:
        for i in report.errors:
            print("ERROR:", i.message)
        raise SystemExit("Data tidak valid; perbaiki dulu (lihat scripts.run_eda).")
    df = df.reset_index(drop=True)

    table = build_feature_table(df)
    train_idx, test_idx = group_stratified_split(df, test_size=0.2, seed=RANDOM_SEED)
    summary = split_summary(df, train_idx, test_idx)
    print(f"Split per perusahaan: latih {summary['n_latih']} baris ({summary['fraud_latih']} fraud), "
          f"uji {summary['n_uji']} baris ({summary['fraud_uji']} fraud), "
          f"perusahaan beririsan: {summary['perusahaan_beririsan']}")

    tr = table.iloc[train_idx]
    res = run_feature_selection(tr[FEATURE_NAMES], tr[TARGET_COLUMN], tr["company_id"],
                                n_boot=n_boot, seed=RANDOM_SEED)
    print(f"Fitur awal {len(FEATURE_NAMES)} -> terpilih {len(res.selected)}: {', '.join(res.selected)}")
    for _, r in res.report[res.report.status == "dibuang"].iterrows():
        print(f"   dibuang {r['fitur']:28s} {r['alasan']}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(PROCESSED_DIR / "features.csv", index=False)
    (PROCESSED_DIR / "feature_selection.json").write_text(
        json.dumps(res.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    (PROCESSED_DIR / "feature_catalog.json").write_text(
        json.dumps(catalog_as_records(), ensure_ascii=False, indent=2), encoding="utf-8")
    (PROCESSED_DIR / "split.json").write_text(json.dumps(
        {"seed": RANDOM_SEED, "metode": "group_stratified_split", "ringkasan": summary,
         "train_idx": train_idx.tolist(), "test_idx": test_idx.tolist()}, indent=1), encoding="utf-8")
    print(f"\nTersimpan di {PROCESSED_DIR}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
