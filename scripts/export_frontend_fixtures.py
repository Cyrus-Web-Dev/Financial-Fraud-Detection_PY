"""Ekspor respons API asli (dari layanan Python) sebagai fixture untuk uji frontend.

    python -m scripts.export_frontend_fixtures

Fixture dipakai oleh tes frontend dan server mock (frontend/tests) sehingga bentuk data yang diuji
SAMA dengan yang dikirim API sungguhan. Hasil: frontend/tests/fixtures/*.json
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd

from src.config import BASE_DIR, SAMPLE_CLEAN_FILE
from src.services.dataset_store import DatasetStore
from src.services.facade import FraudDetectionService
from src.services.history import HistoryRepository

OUT = BASE_DIR / "frontend" / "tests" / "fixtures"


def dump(name: str, obj) -> None:
    (OUT / f"{name}.json").write_text(json.dumps(obj, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"  {name}.json")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        svc = FraudDetectionService(store=DatasetStore(Path(d) / "up"), history=HistoryRepository(Path(d) / "h.db"))
        dump("health", svc.health())
        dump("models", svc.list_models())
        dump("catalog", svc.feature_catalog())
        dump("model_detail_logit", svc.model_detail("logit_mle"))
        up = svc.load_sample("bersih")
        dump("upload_sample", up)
        did = up["dataset_id"]
        dump("preview", svc.dataset_preview(did, rows=60))
        dump("features", svc.dataset_features(did, rows=60))
        dump("predict_dataset", svc.predict_dataset(did, "random_forest", tahun=2023, explain=False))
        rows = pd.read_csv(SAMPLE_CLEAN_FILE).drop(columns=["is_fraud"]).head(6).to_dict("records")
        dump("predict_rows", svc.predict_rows(rows, "random_forest", explain=True))
        base = {r["fitur"]: r["median"] for r in svc.registry.feature_ranges()}
        dump("what_if", svc.what_if("logit_mle", base, {"dsri": base["dsri"] + 0.3}))
        dump("history", svc.history_list(limit=20))
        dump("dataset_id", {"dataset_id": did})
    # hasil analisis tahap sebelumnya disalin apa adanya (dipakai server mock)
    print("selesai ->", OUT)


if __name__ == "__main__":
    main()
