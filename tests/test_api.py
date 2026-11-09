"""Uji end-to-end API dengan TestClient. DILEWATI otomatis bila fastapi/httpx belum terpasang.

Jalankan (setelah `pip install -r requirements.txt`):  python -m tests.test_api   atau   pytest tests/test_api.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from src.config import SAMPLE_CLEAN_FILE
from src.services.dataset_store import DatasetStore
from src.services.facade import FraudDetectionService
from src.services.history import HistoryRepository

try:
    from fastapi.testclient import TestClient
    from src.api.main import create_app
    AVAILABLE = True
except ImportError:          # fastapi / httpx belum terpasang
    AVAILABLE = False


def _client(tmp: str):
    svc = FraudDetectionService(store=DatasetStore(Path(tmp) / "up"), history=HistoryRepository(Path(tmp) / "h.db"))
    return TestClient(create_app(svc))


def _err(resp, status: int, code: str) -> None:
    assert resp.status_code == status, (resp.status_code, resp.text[:300])
    body = resp.json()
    assert body["error"]["kode"] == code, body


def test_info_endpoints_and_headers() -> None:
    if not AVAILABLE:
        print("      (dilewati: fastapi/httpx belum terpasang)"); return
    with tempfile.TemporaryDirectory() as d:
        c = _client(d)
        r = c.get("/api/health"); assert r.status_code == 200 and r.json()["status"] == "ok"
        assert r.headers["x-content-type-options"] == "nosniff" and r.headers["cache-control"] == "no-store"
        assert len(c.get("/api/models").json()["model"]) == 4
        assert len(c.get("/api/features/catalog").json()["fitur"]) == 36
        assert c.get("/api/models/random_forest").status_code == 200
        _err(c.get("/api/models/xyz"), 400, "MODEL_TIDAK_DIKENAL")
        assert c.get("/api/tidak-ada").status_code == 404
        cors = c.get("/api/health", headers={"Origin": "http://localhost:5173"})
        assert cors.headers.get("access-control-allow-origin") == "http://localhost:5173"
        assert "access-control-allow-origin" not in c.get("/api/health", headers={"Origin": "http://jahat.example"}).headers


def test_upload_eda_predict_whatif_history_flow() -> None:
    if not AVAILABLE:
        print("      (dilewati: fastapi/httpx belum terpasang)"); return
    with tempfile.TemporaryDirectory() as d:
        c = _client(d)
        raw = SAMPLE_CLEAN_FILE.read_bytes()
        up = c.post("/api/datasets", files={"file": ("data.csv", raw, "text/csv")})
        assert up.status_code == 200, up.text[:300]
        did = up.json()["dataset_id"]
        assert up.json()["n_baris"] == 3600
        assert c.post("/api/datasets/sample?kind=kotor").status_code == 200
        _err(c.post("/api/datasets/sample?kind=x"), 400, "SAMPEL_TIDAK_DIKENAL")

        pv = c.get(f"/api/datasets/{did}/preview?rows=5&offset=2").json()
        assert len(pv["baris"]) == 5 and pv["total"] == 3600
        _err(c.get(f"/api/datasets/{did}/preview?rows=0"), 422, "INPUT_TIDAK_VALID")
        _err(c.get("/api/datasets/" + "a" * 32 + "/preview"), 404, "DATASET_TIDAK_DITEMUKAN")
        _err(c.get("/api/datasets/NOT-VALID/eda"), 400, "DATASET_ID_TIDAK_VALID")
        assert "benford_per_kelompok" in c.get(f"/api/datasets/{did}/eda").json()
        assert len(c.get(f"/api/datasets/{did}/features?rows=3").json()["baris"]) == 3

        p = c.post("/api/predict/dataset", json={"dataset_id": did, "model": "logreg", "tahun": 2023, "explain": False})
        assert p.status_code == 200 and p.json()["n_baris"] == 600 and 0 < p.json()["ambang"] < 1
        p2 = c.post("/api/predict/dataset", json={"dataset_id": did, "model": "logreg", "threshold": 0.5, "tahun": 2023, "explain": False})
        assert p2.json()["n_ditandai"] < p.json()["n_ditandai"]
        _err(c.post("/api/predict/dataset", json={"dataset_id": did, "threshold": 1.5}), 422, "INPUT_TIDAK_VALID")
        _err(c.post("/api/predict/dataset", json={"dataset_id": "pendek"}), 422, "INPUT_TIDAK_VALID")
        _err(c.post("/api/predict/dataset", json={"dataset_id": did, "model": "isolation_forest"}), 400, "MODEL_TIDAK_DIKENAL")

        rows = pd.read_csv(SAMPLE_CLEAN_FILE).drop(columns=["is_fraud"]).head(2).to_dict("records")
        pr = c.post("/api/predict/rows", json={"rows": rows, "model": "random_forest"})
        assert pr.status_code == 200 and len(pr.json()["hasil"][0]["faktor_utama"]) == 3
        _err(c.post("/api/predict/rows", json={"rows": []}), 422, "INPUT_TIDAK_VALID")
        _err(c.post("/api/predict/rows", json={"rows": [{"company_id": "x"}]}), 422, "DATA_TIDAK_VALID")

        ranges = c.get("/api/models/logit_mle").json()["rentang_fitur"]
        base = {r["fitur"]: r["median"] for r in ranges}
        w = c.post("/api/what-if", json={"model": "logit_mle", "features": base, "overrides": {"dsri": base["dsri"] + 0.3}})
        assert w.status_code == 200 and w.json()["selisih_probabilitas"] > 0
        _err(c.post("/api/what-if", json={"model": "logit_mle", "features": base, "overrides": {"zzz": 1.0}}), 400, "FITUR_TIDAK_DIKENAL")
        _err(c.post("/api/what-if", json={"model": "logit_mle", "features": base, "overrides": {"dsri": "abc"}}), 422, "INPUT_TIDAK_VALID")

        assert c.get("/api/results/evaluation?section=biaya").json()["c_fn"] == 10
        _err(c.get("/api/results/evaluation?section=tidak_ada"), 404, "BAGIAN_TIDAK_ADA")
        _err(c.get("/api/results/xyz"), 404, "JENIS_HASIL_TIDAK_DIKENAL")

        h = c.get("/api/history?limit=3&model=random_forest").json()
        assert h["total"] >= 3 + 600 and len(h["data"]) == 2
        assert c.delete("/api/history").json()["dihapus"] >= 600 and c.get("/api/history").json()["total"] == 0
        assert c.delete(f"/api/datasets/{did}").json() == {"dihapus": did}
        _err(c.delete(f"/api/datasets/{did}"), 404, "DATASET_TIDAK_DITEMUKAN")


def test_upload_rejections() -> None:
    if not AVAILABLE:
        print("      (dilewati: fastapi/httpx belum terpasang)"); return
    with tempfile.TemporaryDirectory() as d:
        c = _client(d)
        _err(c.post("/api/datasets", files={"file": ("a.exe", b"a,b\n1,2", "application/octet-stream")}), 422, "FILE_TIDAK_DAPAT_DIBACA")
        _err(c.post("/api/datasets", files={"file": ("a.csv", b"", "text/csv")}), 422, "FILE_TIDAK_DAPAT_DIBACA")
        csv = pd.read_csv(SAMPLE_CLEAN_FILE).drop(columns=["revenue"]).to_csv(index=False).encode()
        _err(c.post("/api/datasets", files={"file": ("a.csv", csv, "text/csv")}), 422, "DATA_TIDAK_VALID")
        assert c.post("/api/datasets").status_code == 422                       # tanpa file


if __name__ == "__main__":
    if not AVAILABLE:
        print("DILEWATI: fastapi/httpx belum terpasang. Jalankan `pip install -r requirements.txt`.")
    else:
        tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
        for t in tests:
            t(); print(f"OK  {t.__name__}")
        print(f"\n{len(tests)} uji lulus.")
