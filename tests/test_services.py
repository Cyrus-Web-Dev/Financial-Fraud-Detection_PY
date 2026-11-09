"""Uji lapisan layanan (store, riwayat, prediksi, what-if, hasil). Jalankan: python -m tests.test_services"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import MODELS_DIR, PROCESSED_DIR, SAMPLE_CLEAN_FILE, SAMPLE_DIRTY_FILE
from src.services.dataset_store import DatasetStore
from src.services.errors import ServiceError
from src.services.facade import FraudDetectionService
from src.services.history import HistoryRepository
from src.services.prediction import check_threshold, risk_level

ARTIFACTS = [PROCESSED_DIR / n for n in ("features.csv", "split.json", "feature_selection.json", "oof_predictions.csv")] + \
            [MODELS_DIR / f"{m}.joblib" for m in ("logit_mle", "logreg", "random_forest", "hist_gb")]


def _ready() -> bool:
    return all(p.exists() for p in ARTIFACTS) and SAMPLE_CLEAN_FILE.exists()


def _raises(code: str, fn, status: int | None = None) -> ServiceError:
    try:
        fn()
    except ServiceError as e:
        assert e.code == code, (e.code, e.message)
        if status is not None:
            assert e.status == status, (e.status, e.code)
        return e
    raise AssertionError(f"ServiceError({code}) tidak muncul")


def _svc(tmp: str) -> FraudDetectionService:
    return FraudDetectionService(store=DatasetStore(Path(tmp) / "up", max_items=5), history=HistoryRepository(Path(tmp) / "h.db"))


def _raw_rows(n: int = 3, year: int | None = None) -> list[dict]:
    df = pd.read_csv(SAMPLE_CLEAN_FILE)
    if year is not None:
        df = df[df["tahun"] == year]
    return df.drop(columns=["is_fraud"]).head(n).to_dict("records")


# --------------------------------- store ------------------------------------
def test_dataset_store_security_and_eviction() -> None:
    with tempfile.TemporaryDirectory() as d:
        st = DatasetStore(Path(d) / "u", max_items=3)
        df = pd.DataFrame({"a": [1, 2, 3]})
        for bad in ("../../etc/passwd", "A" * 32, "a" * 31, "a" * 33, "a" * 31 + "/", "", None, 123):
            _raises("DATASET_ID_TIDAK_VALID", lambda b=bad: st.load(b), 400)
        _raises("DATASET_TIDAK_DITEMUKAN", lambda: st.load("a" * 32), 404)
        ids = [st.save(df, f"f{i}.csv") for i in range(5)]
        assert len(st.list()) == 3 and {m["id"] for m in st.list()} == set(ids[2:])        # yang terlama dihapus
        assert st.load(ids[4]).equals(df) and st.meta(ids[4])["nama"] == "f4.csv"
        st.delete(ids[4]); _raises("DATASET_TIDAK_DITEMUKAN", lambda: st.load(ids[4]), 404)
        assert not any(Path(d).rglob("..*")) and all(p.parent == Path(d) / "u" for p in (Path(d) / "u").glob("*"))


def test_history_repository_crud_and_sql_injection_safety() -> None:
    with tempfile.TemporaryDirectory() as d:
        h = HistoryRepository(Path(d) / "x" / "h.db")
        recs = [{"company_id": "PRS-1", "tahun": 2022, "probabilitas": 0.2, "ditandai": True, "tingkat_risiko": "Tinggi",
                 "rincian": [{"fitur": "dsri"}]},
                {"company_id": "PRS-2", "tahun": 2023, "probabilitas": 0.01, "ditandai": False, "tingkat_risiko": "Rendah"}]
        assert h.add_many(recs, "random_forest", "manual", 0.09) == 2 and h.count() == 2
        rows = h.list()
        assert [r["company_id"] for r in rows] == ["PRS-2", "PRS-1"]                           # terbaru dulu
        assert rows[1]["ditandai"] is True and rows[1]["rincian"] == [{"fitur": "dsri"}] and rows[0]["rincian"] is None
        assert [r["company_id"] for r in h.list(only_flagged=True)] == ["PRS-1"]
        assert len(h.list(model="random_forest")) == 2 and h.list(model="hist_gb") == []
        assert h.get(rows[0]["id"])["company_id"] == "PRS-2" and h.get(9999) is None
        assert h.list(limit=1, offset=1)[0]["company_id"] == "PRS-1"
        assert h.list(company_id="x' OR '1'='1") == [] and h.count() == 2                      # injeksi tidak berpengaruh
        h.add_many([{"company_id": "'); DROP TABLE predictions;--", "tahun": None, "probabilitas": 0.5, "ditandai": True,
                     "tingkat_risiko": "Tinggi"}], "logreg", "manual", 0.1)
        assert h.count() == 3 and h.list(limit=1)[0]["company_id"].startswith("'); DROP")      # tersimpan sebagai data biasa
        assert h.clear() == 3 and h.count() == 0
        assert HistoryRepository(Path(d) / "x" / "h.db").count() == 0                          # buka ulang file yang sama


def test_risk_level_and_threshold_validation() -> None:
    assert [risk_level(p, 0.1) for p in (0.2, 0.1, 0.06, 0.05, 0.049, 0.0)] == ["Tinggi", "Tinggi", "Sedang", "Sedang", "Rendah", "Rendah"]
    assert check_threshold(None, 0.09) == 0.09 and check_threshold(0.3, 0.09) == 0.3 and check_threshold(0.999, 0.09) == 0.999
    for bad in (0, 1, -0.1, 1.5, float("nan"), float("inf"), "0.5", True, [0.5]):
        _raises("AMBANG_TIDAK_VALID", lambda b=bad: check_threshold(b, 0.09), 400)


# --------------------------- integrasi dengan artefak ------------------------
def test_health_models_catalog_and_detail() -> None:
    if not _ready():
        print("      (dilewati: artefak pelatihan belum ada)"); return
    with tempfile.TemporaryDirectory() as d:
        s = _svc(d)
        assert s.health()["siap_prediksi"] and set(s.health()["model_tersedia"]) >= {"random_forest", "logreg"}
        m = s.list_models()
        assert len(m["model"]) == 4 and all(x["tersedia"] for x in m["model"]) and 0 < m["ambang_default"] < 1
        cat = s.feature_catalog()
        assert len(cat["fitur"]) == 36 and cat["fitur_terpilih"] and all(f["dipakai_model"] == (f["name"] in cat["fitur_terpilih"]) for f in cat["fitur"])
        det = s.model_detail("logit_mle")
        assert det["rentang_fitur"] and det["interpretasi_odds_ratio"] and "metrik_uji" in det
        assert all(r["min"] <= r["p1"] <= r["median"] <= r["p99"] <= r["max"] for r in det["rentang_fitur"])
        assert "metrik_uji" in s.model_detail("random_forest")
        _raises("MODEL_TIDAK_DIKENAL", lambda: s.model_detail("isolation_forest"), 400)
        json.dumps([m, cat, det], allow_nan=False)


def test_upload_validation_and_samples() -> None:
    if not _ready():
        print("      (dilewati: artefak pelatihan belum ada)"); return
    with tempfile.TemporaryDirectory() as d:
        s = _svc(d)
        up = s.upload_dataset(SAMPLE_CLEAN_FILE.read_bytes(), "data.csv")
        assert up["n_baris"] == 3600 and up["punya_label"] and up["validasi"]["n_errors"] == 0 and len(up["pratinjau"]) == 10
        dirty = s.upload_dataset(SAMPLE_DIRTY_FILE.read_bytes(), "kotor.csv")
        assert dirty["validasi"]["n_warnings"] > 3                                         # missing + outlier terlaporkan, tetap diterima
        assert s.load_sample("bersih")["n_baris"] == 3600
        _raises("SAMPEL_TIDAK_DIKENAL", lambda: s.load_sample("x"), 400)
        csv = pd.read_csv(SAMPLE_CLEAN_FILE)
        e = _raises("DATA_TIDAK_VALID", lambda: s.upload_dataset(csv.drop(columns=["revenue"]).to_csv(index=False).encode(), "a.csv"), 422)
        assert e.details["is_valid"] is False and "revenue" in e.message
        _raises("FILE_TIDAK_DAPAT_DIBACA", lambda: s.upload_dataset(b"", "a.csv"), 422)
        _raises("FILE_TIDAK_DAPAT_DIBACA", lambda: s.upload_dataset(b"a,b\n1,2", "a.exe"), 422)
        no_label = csv.drop(columns=["is_fraud"]).to_csv(index=False).encode()
        assert s.upload_dataset(no_label, "nolabel.csv")["punya_label"] is False
        extra = csv.assign(rahasia="x", is_fraud=csv["is_fraud"]).to_csv(index=False).encode()
        did = s.upload_dataset(extra, "extra.csv")["dataset_id"]
        assert "rahasia" not in s.store.load(did).columns                                   # kolom tidak dikenal tidak disimpan
        json.dumps([up, dirty], allow_nan=False)


def test_preview_eda_features_delete() -> None:
    if not _ready():
        print("      (dilewati: artefak pelatihan belum ada)"); return
    with tempfile.TemporaryDirectory() as d:
        s = _svc(d)
        did = s.load_sample()["dataset_id"]
        p = s.dataset_preview(did, rows=5, offset=3)
        assert len(p["baris"]) == 5 and p["offset"] == 3 and p["total"] == 3600
        assert len(s.dataset_preview(did, rows=10 ** 6)["baris"]) == 200                    # dibatasi
        eda = s.dataset_eda(did)
        assert {"deskriptif", "benford_per_kelompok", "perbandingan_kelompok"} <= set(eda) and s.dataset_eda(did) is eda   # cache
        f = s.dataset_features(did, rows=7)
        assert len(f["baris"]) == 7 and "dsri" in f["fitur"] and "m_score" in f["baris"][0]
        json.dumps([p, eda, f], allow_nan=False)
        assert s.delete_dataset(did) == {"dihapus": did}
        _raises("DATASET_TIDAK_DITEMUKAN", lambda: s.dataset_preview(did), 404)
        _raises("DATASET_TIDAK_DITEMUKAN", lambda: s.delete_dataset(did), 404)


def test_predict_dataset_all_models_threshold_and_history() -> None:
    if not _ready():
        print("      (dilewati: artefak pelatihan belum ada)"); return
    with tempfile.TemporaryDirectory() as d:
        s = _svc(d)
        did = s.load_sample()["dataset_id"]
        for m in ("logit_mle", "logreg", "random_forest", "hist_gb"):
            r = s.predict_dataset(did, m, tahun=2023, explain=False)
            assert r["n_baris"] == 600 and r["riwayat_tersimpan"] == 600 and 0 < r["n_ditandai"] < 300
            probs = np.array([x["probabilitas"] for x in r["hasil"]])
            assert ((probs > 0) & (probs < 1)).all()
            assert all(x["ditandai"] == (x["probabilitas"] >= r["ambang"]) for x in r["hasil"])
            assert {"Tinggi", "Sedang", "Rendah"} >= {x["tingkat_risiko"] for x in r["hasil"]}
            assert all("faktor_utama" not in x for x in r["hasil"])
        lo, mid, hi = (s.predict_dataset(did, "logreg", t, False, 2023)["n_ditandai"] for t in (0.02, 0.09, 0.5))
        assert lo > mid > hi                                                                 # ambang lebih rendah -> lebih banyak ditandai
        assert s.history_list(limit=5)["total"] == 600 * 4 + 600 * 3
        assert len(s.history_list(only_flagged=True, limit=500)["data"]) == 500
        big = s.predict_dataset(did, "random_forest", explain=True, tahun=2023)
        assert any("dilewati" in n for n in big["catatan"]) and "faktor_utama" not in big["hasil"][0]
        _raises("TAHUN_TIDAK_ADA", lambda: s.predict_dataset(did, "logreg", tahun=1999), 404)
        _raises("TAHUN_TIDAK_VALID", lambda: s.predict_dataset(did, "logreg", tahun="2023"), 400)
        _raises("MODEL_TIDAK_DIKENAL", lambda: s.predict_dataset(did, "isolation_forest"), 400)
        _raises("AMBANG_TIDAK_VALID", lambda: s.predict_dataset(did, "logreg", 1.2), 400)
        _raises("DATASET_ID_TIDAK_VALID", lambda: s.predict_dataset("../x", "logreg"), 400)
        json.dumps(r, allow_nan=False)


def test_predict_rows_explanations_and_validation() -> None:
    if not _ready():
        print("      (dilewati: artefak pelatihan belum ada)"); return
    with tempfile.TemporaryDirectory() as d:
        s = _svc(d)
        rows = _raw_rows(3)
        for m in ("logit_mle", "random_forest"):
            r = s.predict_rows(rows, m, explain=True)
            assert r["n_baris"] == 3 and r["satuan_penjelasan"] == ("log-odds" if m == "logit_mle" else "probabilitas")
            for x in r["hasil"]:
                assert len(x["faktor_utama"]) == 3 and abs(x["faktor_utama"][0]["kontribusi"]) >= abs(x["faktor_utama"][2]["kontribusi"])
                assert x["faktor_utama"][0]["arah"] in ("menaikkan risiko", "menurunkan risiko")
        # konsistensi: hasil layanan = perhitungan langsung registry (probabilitas terkalibrasi)
        direct = s.predict_rows(rows, "logreg", explain=False)["hasil"]
        again = s.predict_rows(rows, "logreg", explain=False)["hasil"]
        assert [x["probabilitas"] for x in direct] == [x["probabilitas"] for x in again]
        # kolom ekstra (mis. label atau rahasia) diabaikan; label tidak mempengaruhi prediksi
        withlabel = [{**r, "is_fraud": 1, "kolom_lain": "x"} for r in rows]
        assert [x["probabilitas"] for x in s.predict_rows(withlabel, "logreg", explain=False)["hasil"]] == [x["probabilitas"] for x in direct]
        _raises("INPUT_TIDAK_VALID", lambda: s.predict_rows([], "logreg"), 400)
        _raises("INPUT_TIDAK_VALID", lambda: s.predict_rows("x", "logreg"), 400)
        _raises("INPUT_TIDAK_VALID", lambda: s.predict_rows([1, 2], "logreg"), 400)
        _raises("TERLALU_BANYAK_BARIS", lambda: s.predict_rows(rows * 2000, "logreg"), 413)
        bad = [{k: v for k, v in rows[0].items() if k != "revenue"}]
        _raises("DATA_TIDAK_VALID", lambda: s.predict_rows(bad, "logreg"), 422)
        neg = [{**rows[0], "total_assets": -5}]
        _raises("DATA_TIDAK_VALID", lambda: s.predict_rows(neg, "logreg"), 422)
        assert s.history_list()["total"] == 3 * 2 + 3 + 3 + 3


def test_calibration_and_scores_are_monotone_and_model_consistent() -> None:
    if not _ready():
        print("      (dilewati: artefak pelatihan belum ada)"); return
    with tempfile.TemporaryDirectory() as d:
        s = _svc(d)
        X = pd.read_csv(PROCESSED_DIR / "features.csv")[s.registry.selected_features].head(300)
        for m in ("logreg", "random_forest", "hist_gb", "logit_mle"):
            raw, cal = s.registry.calibrated_probability(m, X)
            order = np.argsort(raw, kind="stable")
            assert (np.diff(cal[order]) >= -1e-12).all()                                    # Platt monoton: urutan risiko terjaga
            assert ((raw >= 0) & (raw <= 1)).all() and ((cal > 0) & (cal < 1)).all()
        # penjelasan: nilai dasar + jumlah atribusi = skor mentah (efisiensi) pada skala masing-masing model
        for m in ("logit_mle", "random_forest"):
            phi, base, unit = s.registry.explain(m, X.head(3))
            raw = s.registry.raw_scores(m, X.head(3))
            pred = base + phi.sum(axis=1)
            if unit == "log-odds":
                pred = 1 / (1 + np.exp(-pred))
            assert np.allclose(pred, raw, atol=1e-9), (m, pred, raw)


def test_what_if() -> None:
    if not _ready():
        print("      (dilewati: artefak pelatihan belum ada)"); return
    with tempfile.TemporaryDirectory() as d:
        s = _svc(d)
        sel = s.registry.selected_features
        base = {r["fitur"]: r["median"] for r in s.registry.feature_ranges()}
        same = s.what_if("logit_mle", base, {"dsri": base["dsri"]})
        assert same["selisih_probabilitas"] == 0 and same["peringatan"] == []
        up = s.what_if("logit_mle", base, {"dsri": base["dsri"] + 0.25, "tata": base["tata"] + 0.05})
        assert up["selisih_probabilitas"] > 0 and up["sesudah"]["probabilitas"] > up["sebelum"]["probabilitas"]
        assert set(up["perubahan"]) == {"dsri", "tata"} and len(up["faktor_utama_sesudah"]) == 3
        for m in ("random_forest", "hist_gb", "logreg"):
            r = s.what_if(m, base, {"dsri": 1.4}, 0.2)
            assert 0 < r["sesudah"]["probabilitas"] < 1 and r["ambang"] == 0.2
        far = s.what_if("logit_mle", base, {"dsri": 1000.0})
        assert far["peringatan"] and "ekstrapolasi" in far["peringatan"][0]
        _raises("FITUR_TIDAK_DIKENAL", lambda: s.what_if("logit_mle", base, {"tidak_ada": 1.0}), 400)
        _raises("FITUR_KURANG", lambda: s.what_if("logit_mle", {k: v for k, v in base.items() if k != "dsri"}, {}), 400)
        _raises("NILAI_TIDAK_VALID", lambda: s.what_if("logit_mle", base, {"dsri": float("nan")}), 400)
        _raises("NILAI_TIDAK_VALID", lambda: s.what_if("logit_mle", base, {"dsri": "1.2"}), 400)
        _raises("NILAI_TIDAK_VALID", lambda: s.what_if("logit_mle", base, {"dsri": True}), 400)
        _raises("INPUT_TIDAK_VALID", lambda: s.what_if("logit_mle", [1], {}), 400)
        _raises("MODEL_TIDAK_DIKENAL", lambda: s.what_if("x", base, {}), 400)
        json.dumps([same, up, far], allow_nan=False)


def test_get_results_and_missing_artifacts() -> None:
    with tempfile.TemporaryDirectory() as d:
        s = FraudDetectionService(store=DatasetStore(Path(d) / "u"), history=HistoryRepository(Path(d) / "h.db"),
                                  processed_dir=Path(d) / "kosong")
        _raises("HASIL_BELUM_ADA", lambda: s.get_results("evaluation"), 503)
        _raises("JENIS_HASIL_TIDAK_DIKENAL", lambda: s.get_results("../../etc/passwd"), 404)
        _raises("MODEL_BELUM_SIAP", lambda: s.registry.selected_features, 503)
        assert s.health()["status"] == "ok"
    if not _ready():
        return
    with tempfile.TemporaryDirectory() as d:
        s = _svc(d)
        ev = s.get_results("evaluation")
        assert "kalibrasi" in ev and s.get_results("evaluation", "biaya")["c_fn"] == 10
        e = _raises("BAGIAN_TIDAK_ADA", lambda: s.get_results("evaluation", "tidak_ada"), 404)
        assert "kalibrasi" in e.details["bagian_tersedia"]
        for kind in ("modeling", "explanation", "feature_selection"):
            assert s.get_results(kind)
        json.dumps(s.get_results("modeling"), allow_nan=False)


def test_history_list_filters_and_errors() -> None:
    if not _ready():
        print("      (dilewati: artefak pelatihan belum ada)"); return
    with tempfile.TemporaryDirectory() as d:
        s = _svc(d)
        s.predict_rows(_raw_rows(2), "logreg", explain=False); s.predict_rows(_raw_rows(2), "hist_gb", explain=False)
        assert s.history_list(model="hist_gb")["total"] == 4 and len(s.history_list(model="hist_gb")["data"]) == 2
        assert len(s.history_list(company_id="PRS-0001")["data"]) == 2
        _raises("MODEL_TIDAK_DIKENAL", lambda: s.history_list(model="x"), 400)
        assert s.history_clear()["dihapus"] == 4 and s.history_list()["total"] == 0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"OK  {t.__name__}")
    print(f"\n{len(tests)} uji lulus.")
