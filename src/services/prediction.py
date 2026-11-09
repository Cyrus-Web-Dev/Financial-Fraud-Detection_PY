"""Layanan prediksi: dari laporan keuangan mentah menjadi probabilitas terkalibrasi, keputusan, dan faktor utama."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.config import EXPLAIN_MAX_ROWS, MAX_PREDICT_ROWS
from src.data.validation import validate_dataset
from src.features.pipeline import build_feature_table
from src.features.ratios import FEATURE_CATALOG
from src.services.errors import ServiceError
from src.services.registry import PREDICT_MODELS, ModelRegistry

_LABELS = {f.name: f.label for f in FEATURE_CATALOG}


def risk_level(prob: float, threshold: float) -> str:
    """Tinggi: p >= ambang | Sedang: p >= ambang/2 | Rendah: selainnya."""
    if prob >= threshold:
        return "Tinggi"
    return "Sedang" if prob >= threshold / 2.0 else "Rendah"


def check_threshold(threshold: float | None, default: float) -> float:
    if threshold is None:
        return float(default)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) \
            or not 0.0 < float(threshold) < 1.0:
        raise ServiceError("AMBANG_TIDAK_VALID", "Ambang harus berupa angka di antara 0 dan 1 (eksklusif).", 400)
    return float(threshold)


def _top_factors(phi_row: np.ndarray, x_row: pd.Series, names: list[str], unit: str, top_n: int = 3) -> list[dict[str, Any]]:
    order = np.argsort(-np.abs(phi_row))[:top_n]
    return [{"fitur": names[j], "label": _LABELS.get(names[j], names[j]), "nilai": float(x_row[names[j]]),
             "kontribusi": float(phi_row[j]), "arah": "menaikkan risiko" if phi_row[j] > 0 else "menurunkan risiko",
             "satuan": unit} for j in order]


class PredictionService:
    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry

    # -------------------------------------------------------------- utilitas
    def check_model(self, model: str) -> None:
        if model not in PREDICT_MODELS:
            raise ServiceError("MODEL_TIDAK_DIKENAL", f"Model '{model}' tidak dikenal.", 400, {"pilihan": list(PREDICT_MODELS)})

    def validate_raw(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validasi data mentah untuk prediksi (label tidak wajib). Error validasi -> ServiceError."""
        clean, report = validate_dataset(df, require_target=False)
        if not report.is_valid:
            raise ServiceError("DATA_TIDAK_VALID", "Data tidak lolos validasi: " + "; ".join(i.message for i in report.errors[:3]),
                               422, report.to_dict())
        return clean.reset_index(drop=True)

    # -------------------------------------------------------------- prediksi
    def predict_frame(self, clean: pd.DataFrame, model: str, threshold: float | None = None, explain: bool = True
                      ) -> dict[str, Any]:
        """Prediksi untuk data MENTAH yang sudah divalidasi (``validate_raw``)."""
        self.check_model(model)
        thr = check_threshold(threshold, self.registry.default_threshold())
        if len(clean) == 0:
            raise ServiceError("DATA_KOSONG", "Tidak ada baris untuk diprediksi.", 400)
        if len(clean) > MAX_PREDICT_ROWS:
            raise ServiceError("TERLALU_BANYAK_BARIS", f"Maksimal {MAX_PREDICT_ROWS} baris per permintaan "
                               f"(diberikan {len(clean)}).", 413)
        table = build_feature_table(clean)
        sel = self.registry.selected_features
        X = table[sel]
        raw, prob = self.registry.calibrated_probability(model, X)

        notes: list[str] = []
        phi = base = None
        unit = PREDICT_MODELS[model]["satuan_penjelasan"]
        if explain and len(clean) <= EXPLAIN_MAX_ROWS:
            phi, base, unit = self.registry.explain(model, X)
        elif explain:
            notes.append(f"Penjelasan per baris dilewati karena jumlah baris > {EXPLAIN_MAX_ROWS} (biaya komputasi).")

        rows = []
        for i in range(len(clean)):
            p = float(prob[i])
            row = {"company_id": str(clean.loc[i, "company_id"]), "tahun": int(clean.loc[i, "tahun"]),
                   "sektor": str(clean.loc[i, "sektor"]), "skor_model": float(raw[i]), "probabilitas": p,
                   "ditandai": bool(p >= thr), "tingkat_risiko": risk_level(p, thr),
                   "m_score": None if pd.isna(table.loc[i, "m_score"]) else float(table.loc[i, "m_score"]),
                   "m_flag": None if pd.isna(table.loc[i, "m_flag"]) else bool(table.loc[i, "m_flag"]),
                   "altman_z2_zona": table.loc[i, "altman_z2_zone"] if pd.notna(table.loc[i, "altman_z2_zone"]) else None}
            if phi is not None:
                row["faktor_utama"] = _top_factors(phi[i], X.iloc[i], sel, unit)
            rows.append(row)
        flagged = int(sum(r["ditandai"] for r in rows))
        return {"model": model, "label_model": PREDICT_MODELS[model]["label"], "ambang": thr, "n_baris": len(rows),
                "n_ditandai": flagged, "persen_ditandai": flagged / len(rows), "nilai_dasar_penjelasan": base,
                "satuan_penjelasan": unit, "catatan": notes, "hasil": rows}

    # -------------------------------------------------------------- what-if
    def what_if(self, model: str, features: dict[str, float], overrides: dict[str, float], threshold: float | None = None
                ) -> dict[str, Any]:
        """Dampak perubahan nilai fitur (rasio) terhadap prediksi: probabilitas sebelum, sesudah, dan selisih."""
        self.check_model(model)
        thr = check_threshold(threshold, self.registry.default_threshold())
        sel = self.registry.selected_features
        if not isinstance(features, dict) or not isinstance(overrides, dict):
            raise ServiceError("INPUT_TIDAK_VALID", "features dan overrides harus berupa objek.", 400)
        unknown = [k for k in list(features) + list(overrides) if k not in sel]
        if unknown:
            raise ServiceError("FITUR_TIDAK_DIKENAL", f"Fitur tidak dikenal: {', '.join(sorted(set(map(str, unknown))))}", 400,
                               {"fitur_valid": sel})
        absent = [f for f in sel if f not in features]
        if absent:
            raise ServiceError("FITUR_KURANG", f"Nilai fitur berikut wajib diisi: {', '.join(absent)}", 400)
        for k, v in {**features, **overrides}.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ServiceError("NILAI_TIDAK_VALID", f"Nilai fitur '{k}' harus berupa angka hingga.", 400)

        base_row = {f: float(features[f]) for f in sel}
        new_row = {**base_row, **{k: float(v) for k, v in overrides.items()}}
        Xb, Xn = pd.DataFrame([base_row]), pd.DataFrame([new_row])
        _, p0 = self.registry.calibrated_probability(model, Xb)
        _, p1 = self.registry.calibrated_probability(model, Xn)
        phi, base, unit = self.registry.explain(model, Xn)

        ranges = {r["fitur"]: r for r in self.registry.feature_ranges()}
        warnings = [f"Nilai '{k}' ({float(v):.4g}) berada di luar rentang data latih [{ranges[k]['min']:.4g}, {ranges[k]['max']:.4g}]; "
                    "prediksi merupakan ekstrapolasi dan kurang dapat dipercaya."
                    for k, v in overrides.items() if not ranges[k]["min"] <= float(v) <= ranges[k]["max"]]
        return {"model": model, "ambang": thr,
                "sebelum": {"probabilitas": float(p0[0]), "ditandai": bool(p0[0] >= thr), "tingkat_risiko": risk_level(float(p0[0]), thr)},
                "sesudah": {"probabilitas": float(p1[0]), "ditandai": bool(p1[0] >= thr), "tingkat_risiko": risk_level(float(p1[0]), thr)},
                "selisih_probabilitas": float(p1[0] - p0[0]), "perubahan": {k: {"dari": base_row[k], "ke": new_row[k]} for k in overrides},
                "faktor_utama_sesudah": _top_factors(phi[0], Xn.iloc[0], sel, unit), "peringatan": warnings}
