"""Fasad layanan: SATU method untuk SETIAP endpoint API.

Seluruh validasi masukan, aturan bisnis, dan pemetaan error ada di sini (Python murni), sehingga lapisan HTTP
(``src/api/main.py``) hanya memetakan rute ke method ini. Semua keluaran dijamin JSON-aman (tanpa NaN/inf).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    HISTORY_DB_FILE, MAX_PREDICT_ROWS, MODELS_DIR, PROCESSED_DIR, SAMPLE_CLEAN_FILE, SAMPLE_DIRTY_FILE, UPLOAD_DIR,
)
from src.data.loader import DataLoadError, load_dataset
from src.data.schema import REQUIRED_COLUMNS_FOR_PREDICTION, TARGET_COLUMN
from src.data.validation import validate_dataset
from src.evaluation.eda_summary import build_eda_summary
from src.features.pipeline import build_feature_table
from src.features.ratios import FEATURE_NAMES, catalog_as_records
from src.reporting.builder import ReportData, build_report_pdf
from src.services.dataset_store import DatasetStore
from src.services.errors import ServiceError
from src.services.history import HistoryRepository
from src.services.prediction import PredictionService
from src.services.registry import PREDICT_MODELS, ModelRegistry
from src.utils import to_jsonable

RESULT_FILES: dict[str, str] = {"modeling": "modeling_results.json", "evaluation": "evaluation_results.json",
                                "explanation": "explanation_results.json", "feature_selection": "feature_selection.json",
                                "eda_sample": "eda_summary.json"}
SAMPLES = {"bersih": SAMPLE_CLEAN_FILE, "kotor": SAMPLE_DIRTY_FILE}
MAX_PREVIEW_ROWS = 200


def _jsonable(obj: Any) -> Any:
    return to_jsonable(obj)


class FraudDetectionService:
    def __init__(self, registry: ModelRegistry | None = None, store: DatasetStore | None = None,
                 history: HistoryRepository | None = None, processed_dir: Path | str = PROCESSED_DIR) -> None:
        self.processed_dir = Path(processed_dir)
        self.registry = registry or ModelRegistry(MODELS_DIR, self.processed_dir)
        self.store = store or DatasetStore(UPLOAD_DIR)
        self.history = history or HistoryRepository(HISTORY_DB_FILE)
        self.predictor = PredictionService(self.registry)
        self._eda_cache: dict[str, dict] = {}
        self._results_cache: dict[str, tuple[float, Any]] = {}

    # ------------------------------------------------------------ informasi umum
    def health(self) -> dict[str, Any]:
        available = self.registry.available_models()
        return _jsonable({"status": "ok", "siap_prediksi": len(available) > 0, "model_tersedia": available,
                          "jumlah_riwayat": self.history.count()})

    def list_models(self) -> dict[str, Any]:
        avail = set(self.registry.available_models())
        models = [{"nama": n, **meta, "tersedia": n in avail} for n, meta in PREDICT_MODELS.items()]
        return _jsonable({"model": models, "ambang_default": self.registry.default_threshold() if avail else None})

    def model_detail(self, name: str) -> dict[str, Any]:
        info = self.registry.model_info(name)
        info["rentang_fitur"] = self.registry.feature_ranges()
        if name == "logit_mle":
            try:
                info["interpretasi_odds_ratio"] = self.get_results("explanation", "interpretasi_odds_ratio")
            except ServiceError:
                info["interpretasi_odds_ratio"] = None
        return _jsonable(info)

    def feature_catalog(self) -> dict[str, Any]:
        selected = set(self.registry.selected_features) if self.registry.available_models() else set()
        return _jsonable({"fitur": [{**r, "dipakai_model": r["name"] in selected} for r in catalog_as_records()],
                          "fitur_terpilih": sorted(selected)})

    # ------------------------------------------------------------ dataset
    def upload_dataset(self, content: bytes, filename: str, synthetic: bool = False) -> dict[str, Any]:
        """Baca, validasi, dan simpan dataset. Dataset dengan ERROR tidak disimpan (laporan dikembalikan sebagai 422)."""
        try:
            raw = load_dataset(content, filename=filename)
        except DataLoadError as exc:
            raise ServiceError("FILE_TIDAK_DAPAT_DIBACA", str(exc), 422) from exc
        clean, report = validate_dataset(raw, require_target=False)
        if not report.is_valid:
            raise ServiceError("DATA_TIDAK_VALID", "Data tidak lolos validasi: " + "; ".join(i.message for i in report.errors[:3]),
                               422, _jsonable(report.to_dict()))
        keep = [c for c in clean.columns if c in set(REQUIRED_COLUMNS_FOR_PREDICTION) | {TARGET_COLUMN}]
        clean = clean[keep].reset_index(drop=True)
        has_target = TARGET_COLUMN in clean.columns and clean[TARGET_COLUMN].notna().any()
        dataset_id = self.store.save(clean, filename, {"punya_label": bool(has_target), "sintetis": bool(synthetic)})
        return _jsonable({"dataset_id": dataset_id, "nama": filename, "n_baris": len(clean), "punya_label": bool(has_target),
                          "validasi": report.to_dict(), "pratinjau": clean.head(10)})

    def load_sample(self, kind: str = "bersih") -> dict[str, Any]:
        if kind not in SAMPLES:
            raise ServiceError("SAMPEL_TIDAK_DIKENAL", "Pilihan sampel: 'bersih' atau 'kotor'.", 400)
        path = SAMPLES[kind]
        if not path.exists():
            raise ServiceError("SAMPEL_TIDAK_ADA", "File sampel belum dibuat. Jalankan scripts.generate_sample_data.", 503)
        return self.upload_dataset(path.read_bytes(), path.name, synthetic=True)

    def dataset_preview(self, dataset_id: str, rows: int = 20, offset: int = 0) -> dict[str, Any]:
        df = self.store.load(dataset_id)
        rows, offset = max(1, min(int(rows), MAX_PREVIEW_ROWS)), max(0, int(offset))
        return _jsonable({"meta": self.store.meta(dataset_id), "offset": offset, "total": len(df),
                          "baris": df.iloc[offset:offset + rows]})

    def delete_dataset(self, dataset_id: str) -> dict[str, Any]:
        self.store.load(dataset_id)            # memastikan ada (404 bila tidak)
        self.store.delete(dataset_id)
        self._eda_cache.pop(dataset_id, None)
        return {"dihapus": dataset_id}

    def dataset_eda(self, dataset_id: str) -> dict[str, Any]:
        if dataset_id not in self._eda_cache:
            df = self.store.load(dataset_id)
            clean, _ = validate_dataset(df, require_target=False)
            self._eda_cache[dataset_id] = build_eda_summary(clean)
        return self._eda_cache[dataset_id]

    def dataset_features(self, dataset_id: str, rows: int = 20) -> dict[str, Any]:
        clean, _ = validate_dataset(self.store.load(dataset_id), require_target=False)
        table = build_feature_table(clean)
        rows = max(1, min(int(rows), MAX_PREVIEW_ROWS))
        return _jsonable({"fitur": FEATURE_NAMES, "baris": table.head(rows)})

    # ------------------------------------------------------------ prediksi
    def predict_dataset(self, dataset_id: str, model: str, threshold: float | None = None, explain: bool = True,
                        tahun: int | None = None) -> dict[str, Any]:
        """Prediksi seluruh dataset (atau satu ``tahun``). Penjelasan per baris hanya bila baris <= EXPLAIN_MAX_ROWS."""
        df = self.store.load(dataset_id)
        if tahun is not None:
            if isinstance(tahun, bool) or not isinstance(tahun, int):
                raise ServiceError("TAHUN_TIDAK_VALID", "tahun harus bilangan bulat.", 400)
            df = df[df["tahun"] == tahun]
            if df.empty:
                raise ServiceError("TAHUN_TIDAK_ADA", f"Tidak ada data untuk tahun {tahun}.", 404)
        clean = self.predictor.validate_raw(df.drop(columns=[TARGET_COLUMN], errors="ignore"))
        result = self.predictor.predict_frame(clean, model, threshold, explain)
        saved = self.history.add_many(result["hasil"], model, f"dataset:{dataset_id[:8]}", result["ambang"])
        return _jsonable({**result, "dataset_id": dataset_id, "riwayat_tersimpan": saved})

    def predict_rows(self, rows: list[dict[str, Any]], model: str, threshold: float | None = None, explain: bool = True
                     ) -> dict[str, Any]:
        if not isinstance(rows, list) or not rows or not all(isinstance(r, dict) for r in rows):
            raise ServiceError("INPUT_TIDAK_VALID", "rows harus berupa daftar objek yang tidak kosong.", 400)
        if len(rows) > MAX_PREDICT_ROWS:
            raise ServiceError("TERLALU_BANYAK_BARIS", f"Maksimal {MAX_PREDICT_ROWS} baris per permintaan.", 413)
        allowed = set(REQUIRED_COLUMNS_FOR_PREDICTION)
        df = pd.DataFrame([{k: v for k, v in r.items() if k in allowed} for r in rows])
        clean = self.predictor.validate_raw(df)
        result = self.predictor.predict_frame(clean, model, threshold, explain)
        saved = self.history.add_many(result["hasil"], model, "manual", result["ambang"])
        return _jsonable({**result, "riwayat_tersimpan": saved})

    def what_if(self, model: str, features: dict[str, float], overrides: dict[str, float], threshold: float | None = None
                ) -> dict[str, Any]:
        return _jsonable(self.predictor.what_if(model, features, overrides, threshold))

    # ------------------------------------------------------------ laporan PDF
    def generate_report(self, dataset_id: str, model: str, threshold: float | None = None, tahun: int | None = None,
                        top_n: int = 20) -> tuple[bytes, str]:
        """Buat laporan PDF untuk dataset: (isi PDF, nama berkas). Faktor utama dihitung untuk ``top_n`` observasi teratas."""
        if isinstance(top_n, bool) or not isinstance(top_n, int) or not 1 <= top_n <= 50:
            raise ServiceError("TOP_N_TIDAK_VALID", "top_n harus bilangan bulat 1-50.", 400)
        meta = self.store.meta(dataset_id)
        df = self.store.load(dataset_id)
        if tahun is not None:
            if isinstance(tahun, bool) or not isinstance(tahun, int):
                raise ServiceError("TAHUN_TIDAK_VALID", "tahun harus bilangan bulat.", 400)
            df = df[df["tahun"] == tahun]
            if df.empty:
                raise ServiceError("TAHUN_TIDAK_ADA", f"Tidak ada data untuk tahun {tahun}.", 404)
        raw = df.drop(columns=[TARGET_COLUMN], errors="ignore")
        clean = self.predictor.validate_raw(raw)
        result = self.predictor.predict_frame(clean, model, threshold, explain=False)

        # Penjelasan Shapley hanya untuk top_n observasi teratas (mahal bila dilakukan untuk semua baris).
        top = sorted(result["hasil"], key=lambda r: -r["probabilitas"])[:top_n]
        keys = {(r["company_id"], r["tahun"]) for r in top}
        mask = [(str(c), int(y)) in keys for c, y in zip(clean["company_id"], clean["tahun"])]
        explained = self.predictor.predict_frame(clean[mask].reset_index(drop=True), model, result["ambang"], explain=True)
        factors = {(r["company_id"], r["tahun"]): r.get("faktor_utama") for r in explained["hasil"]}
        for r in result["hasil"]:
            f = factors.get((r["company_id"], r["tahun"]))
            if f:
                r["faktor_utama"] = f

        full, report = validate_dataset(df, require_target=False)
        def optional(kind: str) -> Any:
            try:
                return self.get_results(kind)
            except ServiceError:
                return None
        data = ReportData(
            dataset={"nama": meta.get("nama", ""), "n_baris": meta.get("n_baris", len(df)), "punya_label": bool(meta.get("punya_label")),
                     "validasi": _jsonable(report.to_dict())},
            prediction=_jsonable(result), eda=_jsonable(self.dataset_eda(dataset_id)), evaluation=optional("evaluation"),
            explanation=optional("explanation"), modeling=optional("modeling"), model_info=_jsonable(self.registry.model_info(model)),
            top_n=top_n, synthetic=bool(meta.get("sintetis")), generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
            scope=f"tahun {tahun}" if tahun is not None else "")
        pdf = build_report_pdf(data)
        return pdf, f"laporan_fraud_{model}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"

    # ------------------------------------------------------------ hasil analisis
    def get_results(self, kind: str, section: str | None = None) -> Any:
        """Hasil analisis hasil pelatihan (modeling/evaluation/explanation/...). ``section`` memilih satu kunci tingkat atas."""
        if kind not in RESULT_FILES:
            raise ServiceError("JENIS_HASIL_TIDAK_DIKENAL", f"Jenis hasil tidak dikenal. Pilihan: {', '.join(RESULT_FILES)}", 404)
        path = self.processed_dir / RESULT_FILES[kind]
        if not path.exists():
            raise ServiceError("HASIL_BELUM_ADA", f"Hasil '{kind}' belum dibuat. Jalankan skrip tahap terkait.", 503)
        mtime = path.stat().st_mtime
        cached = self._results_cache.get(kind)
        if cached is None or cached[0] != mtime:
            with open(path, encoding="utf-8") as fh:
                cached = (mtime, json.load(fh))
            self._results_cache[kind] = cached
        data = cached[1]
        if section is None:
            return data if isinstance(data, dict) else {"data": data}
        if not isinstance(data, dict) or section not in data:
            raise ServiceError("BAGIAN_TIDAK_ADA", f"Bagian '{section}' tidak ada pada hasil '{kind}'.", 404,
                               {"bagian_tersedia": list(data) if isinstance(data, dict) else []})
        return data[section]

    # ------------------------------------------------------------ riwayat
    def history_list(self, limit: int = 50, offset: int = 0, model: str | None = None, company_id: str | None = None,
                     only_flagged: bool = False) -> dict[str, Any]:
        if model is not None and model not in PREDICT_MODELS:
            raise ServiceError("MODEL_TIDAK_DIKENAL", f"Model '{model}' tidak dikenal.", 400)
        return _jsonable({"total": self.history.count(),
                          "data": self.history.list(limit, offset, model, company_id, only_flagged)})

    def history_clear(self) -> dict[str, Any]:
        return {"dihapus": self.history.clear()}
