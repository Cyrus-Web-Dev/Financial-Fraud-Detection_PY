"""Lapisan HTTP (FastAPI) -- sengaja tipis.

Setiap endpoint hanya memanggil SATU method pada ``FraudDetectionService`` (``src/services/facade.py``). Seluruh
validasi dan logika bisnis ada di layanan dan sudah diuji. Jalankan:

    uvicorn src.api.main:app --reload --port 8000

Dokumentasi interaktif: http://localhost:8000/docs
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import CORS_ORIGINS, MAX_PREDICT_ROWS, MAX_UPLOAD_MB
from src.services.errors import ServiceError
from src.services.facade import FraudDetectionService
from src.utils import to_jsonable

log = logging.getLogger("fraud.api")
DEFAULT_MODEL = "random_forest"
FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"


# --------------------------------------------------------------------------- skema permintaan
class PredictDatasetRequest(BaseModel):
    dataset_id: str = Field(..., min_length=32, max_length=32, description="id dari POST /api/datasets")
    model: str = Field(DEFAULT_MODEL, description="logit_mle | logreg | random_forest | hist_gb")
    threshold: Optional[float] = Field(None, gt=0, lt=1, description="ambang pada probabilitas terkalibrasi")
    explain: bool = True
    tahun: Optional[int] = Field(None, description="prediksi hanya untuk tahun ini")


class PredictRowsRequest(BaseModel):
    rows: List[Dict[str, Any]] = Field(..., min_length=1, max_length=MAX_PREDICT_ROWS)
    model: str = DEFAULT_MODEL
    threshold: Optional[float] = Field(None, gt=0, lt=1)
    explain: bool = True


class ReportRequest(BaseModel):
    dataset_id: str = Field(..., min_length=32, max_length=32)
    model: str = DEFAULT_MODEL
    threshold: Optional[float] = Field(None, gt=0, lt=1)
    tahun: Optional[int] = None
    top_n: int = Field(20, ge=1, le=50, description="jumlah observasi teratas yang dijelaskan di laporan")


class WhatIfRequest(BaseModel):
    model: str = DEFAULT_MODEL
    features: Dict[str, float] = Field(..., description="nilai awal SELURUH fitur model")
    overrides: Dict[str, float] = Field(..., description="fitur yang diubah beserta nilai barunya")
    threshold: Optional[float] = Field(None, gt=0, lt=1)


def _ok(data: Any) -> JSONResponse:
    return JSONResponse(content=to_jsonable(data))


def create_app(service: Optional[FraudDetectionService] = None) -> FastAPI:
    """Bangun aplikasi. ``service`` dapat disuntikkan (untuk pengujian); default dibuat dari konfigurasi."""
    svc = service or FraudDetectionService()
    app = FastAPI(title="Sistem Deteksi Financial Fraud", version="1.0.0",
                  description="API deteksi indikasi fraud laporan keuangan (skripsi Statistika).")
    app.add_middleware(CORSMiddleware, allow_origins=list(CORS_ORIGINS), allow_methods=["GET", "POST", "DELETE"],
                       allow_headers=["*"], expose_headers=["Content-Disposition"])

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        return response

    # ----------------------------------------------------------------- penanganan error
    @app.exception_handler(ServiceError)
    async def service_error_handler(request: Request, exc: ServiceError):
        return JSONResponse(status_code=exc.status, content={"error": to_jsonable(exc.to_dict())})

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={"error": {
            "kode": "INPUT_TIDAK_VALID", "pesan": "Format permintaan tidak valid.", "rincian": jsonable_encoder(exc.errors())}})

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception):
        log.exception("Kesalahan tak terduga pada %s", request.url.path)   # detail hanya di log server
        return JSONResponse(status_code=500, content={"error": {
            "kode": "KESALAHAN_INTERNAL", "pesan": "Terjadi kesalahan pada server."}})

    # ----------------------------------------------------------------- informasi
    @app.get("/api/health")
    def health():
        return _ok(svc.health())

    @app.get("/api/models")
    def list_models():
        return _ok(svc.list_models())

    @app.get("/api/models/{name}")
    def model_detail(name: str):
        return _ok(svc.model_detail(name))

    @app.get("/api/features/catalog")
    def feature_catalog():
        return _ok(svc.feature_catalog())

    # ----------------------------------------------------------------- dataset
    @app.post("/api/datasets")
    def upload_dataset(file: UploadFile = File(...)):
        limit = MAX_UPLOAD_MB * 1024 * 1024
        chunks, total = [], 0
        while True:
            chunk = file.file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise ServiceError("FILE_TERLALU_BESAR", f"Ukuran file melebihi batas {MAX_UPLOAD_MB} MB.", 413)
            chunks.append(chunk)
        return _ok(svc.upload_dataset(b"".join(chunks), file.filename or ""))

    @app.post("/api/datasets/sample")
    def load_sample(kind: str = Query("bersih", description="bersih | kotor")):
        return _ok(svc.load_sample(kind))

    @app.get("/api/datasets/{dataset_id}/preview")
    def dataset_preview(dataset_id: str, rows: int = Query(20, ge=1, le=200), offset: int = Query(0, ge=0)):
        return _ok(svc.dataset_preview(dataset_id, rows, offset))

    @app.get("/api/datasets/{dataset_id}/eda")
    def dataset_eda(dataset_id: str):
        return _ok(svc.dataset_eda(dataset_id))

    @app.get("/api/datasets/{dataset_id}/features")
    def dataset_features(dataset_id: str, rows: int = Query(20, ge=1, le=200)):
        return _ok(svc.dataset_features(dataset_id, rows))

    @app.delete("/api/datasets/{dataset_id}")
    def delete_dataset(dataset_id: str):
        return _ok(svc.delete_dataset(dataset_id))

    # ----------------------------------------------------------------- prediksi
    @app.post("/api/predict/dataset")
    def predict_dataset(req: PredictDatasetRequest):
        return _ok(svc.predict_dataset(req.dataset_id, req.model, req.threshold, req.explain, req.tahun))

    @app.post("/api/predict/rows")
    def predict_rows(req: PredictRowsRequest):
        return _ok(svc.predict_rows(req.rows, req.model, req.threshold, req.explain))

    @app.post("/api/what-if")
    def what_if(req: WhatIfRequest):
        return _ok(svc.what_if(req.model, req.features, req.overrides, req.threshold))

    @app.post("/api/report")
    def make_report(req: ReportRequest):
        pdf, filename = svc.generate_report(req.dataset_id, req.model, req.threshold, req.tahun, req.top_n)
        return Response(content=pdf, media_type="application/pdf",
                        headers={"Content-Disposition": f'attachment; filename="{filename}"', "Cache-Control": "no-store"})

    # ----------------------------------------------------------------- hasil analisis dan riwayat
    @app.get("/api/results/{kind}")
    def get_results(kind: str, section: Optional[str] = Query(None)):
        return _ok(svc.get_results(kind, section))

    @app.get("/api/history")
    def history_list(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0), model: Optional[str] = None,
                     company_id: Optional[str] = None, only_flagged: bool = False):
        return _ok(svc.history_list(limit, offset, model, company_id, only_flagged))

    @app.delete("/api/history")
    def history_clear():
        return _ok(svc.history_clear())

    # Sajikan frontend hasil `npm run build` (bila ada) dari proses yang sama. Dipasang TERAKHIR agar rute /api diprioritaskan.
    if FRONTEND_DIST.is_dir():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=False)
