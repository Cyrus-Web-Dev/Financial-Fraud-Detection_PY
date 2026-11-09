"""Buat laporan PDF dari file data (tanpa menjalankan server).

    python -m scripts.make_report                              # data sampel, Random Forest
    python -m scripts.make_report data/saya.csv --model logit_mle --tahun 2023 --top-n 30
    python -m scripts.make_report --threshold 0.15 --out reports/laporan_saya.pdf

Prasyarat: tahap pelatihan sudah dijalankan (lihat scripts.run_pipeline).
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from src.config import REPORTS_DIR
from src.services.dataset_store import DatasetStore
from src.services.facade import FraudDetectionService
from src.services.history import HistoryRepository


def main(path: str | None = None, model: str = "random_forest", threshold: float | None = None, tahun: int | None = None,
         top_n: int = 20, out: str | None = None) -> Path:
    with tempfile.TemporaryDirectory() as d:
        svc = FraudDetectionService(store=DatasetStore(Path(d) / "u"), history=HistoryRepository(Path(d) / "h.db"))
        if path:
            src = Path(path)
            up = svc.upload_dataset(src.read_bytes(), src.name)
        else:
            up = svc.load_sample("bersih")
        pdf, name = svc.generate_report(up["dataset_id"], model, threshold, tahun, top_n)
    target = Path(out) if out else REPORTS_DIR / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(pdf)
    print(f"Laporan tersimpan: {target} ({len(pdf) / 1024:.0f} KB)")
    return target


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", help="file CSV/XLSX (default: data sampel)")
    ap.add_argument("--model", default="random_forest", choices=["logit_mle", "logreg", "random_forest", "hist_gb"])
    ap.add_argument("--threshold", type=float, default=None, help="ambang pada probabilitas terkalibrasi (default: ambang Bayes)")
    ap.add_argument("--tahun", type=int, default=None)
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    main(a.path, a.model, a.threshold, a.tahun, a.top_n, a.out)
