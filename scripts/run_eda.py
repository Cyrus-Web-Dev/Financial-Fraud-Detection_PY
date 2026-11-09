"""Jalankan validasi + EDA pada sebuah file data dan simpan ringkasannya sebagai JSON.

    python -m scripts.run_eda                       # memakai data sampel
    python -m scripts.run_eda path/ke/data.csv      # memakai data sendiri
"""
from __future__ import annotations

import json
import sys

from src.config import PROCESSED_DIR, SAMPLE_CLEAN_FILE
from src.data.loader import load_dataset
from src.data.validation import validate_dataset
from src.evaluation.eda_summary import build_eda_summary


def main(path: str | None = None) -> None:
    source = path or SAMPLE_CLEAN_FILE
    df = load_dataset(source)
    clean, report = validate_dataset(df, require_target=True)

    print(f"Validasi: {'LULUS' if report.is_valid else 'GAGAL'} | "
          f"{len(report.errors)} error, {len(report.warnings)} peringatan")
    for issue in report.issues:
        print(f"  [{issue.severity.upper():7s}] {issue.message}")
    if not report.is_valid:
        raise SystemExit("Perbaiki error di atas sebelum melanjutkan.")

    summary = build_eda_summary(clean)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "eda_summary.json"
    out.write_text(json.dumps({"validasi": report.to_dict(), **summary}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\nRingkasan EDA disimpan: {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
