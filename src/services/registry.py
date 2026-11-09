"""Registry model: memuat model terlatih, kalibrator probabilitas, dan penjelas secara lazy (dengan cache).

Probabilitas yang ditampilkan ke pengguna adalah probabilitas TERKALIBRASI (Platt scaling yang dilatih pada
prediksi out-of-fold data latih, lihat Tahap 5), bukan skor mentah model. Ambang keputusan berlaku pada
probabilitas terkalibrasi ini.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, PROCESSED_DIR, RANDOM_SEED
from src.data.schema import TARGET_COLUMN
from src.evaluation.calibration import PlattCalibrator
from src.evaluation.thresholds import bayes_threshold
from src.explanation.shapley import ExactShapleyExplainer, linear_shap, model_scorer
from src.features.ratios import FEATURE_CATALOG
from src.models.training import load_model_bundle
from src.services.errors import ServiceError

PREDICT_MODELS: dict[str, dict[str, str]] = {
    "logit_mle": {"label": "Regresi Logistik (inferensial)", "jenis": "klasik", "satuan_penjelasan": "log-odds"},
    "logreg": {"label": "Regresi Logistik (L2)", "jenis": "klasik", "satuan_penjelasan": "log-odds"},
    "random_forest": {"label": "Random Forest", "jenis": "ml", "satuan_penjelasan": "probabilitas"},
    "hist_gb": {"label": "Gradient Boosting", "jenis": "ml", "satuan_penjelasan": "probabilitas"},
}
N_BACKGROUND = 40


def _read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


class ModelRegistry:
    def __init__(self, models_dir: Path | str = MODELS_DIR, processed_dir: Path | str = PROCESSED_DIR,
                 seed: int = RANDOM_SEED) -> None:
        self.models_dir, self.processed_dir, self.seed = Path(models_dir), Path(processed_dir), seed
        self._lock = threading.RLock()
        self._ctx: dict[str, Any] | None = None
        self._models: dict[str, Any] = {}
        self._calibrators: dict[str, PlattCalibrator] = {}
        self._explainers: dict[str, Any] = {}

    # ------------------------------------------------------------- konteks
    def _context(self) -> dict[str, Any]:
        with self._lock:
            if self._ctx is not None:
                return self._ctx
            needed = ["features.csv", "split.json", "feature_selection.json"]
            missing = [n for n in needed if not (self.processed_dir / n).exists()]
            if missing:
                raise ServiceError("MODEL_BELUM_SIAP", "Artefak pelatihan belum ada. Jalankan scripts.run_feature_engineering "
                                   "dan scripts.run_modeling terlebih dahulu.", 503, {"hilang": missing})
            table = pd.read_csv(self.processed_dir / "features.csv")
            split = _read_json(self.processed_dir / "split.json")
            selected = _read_json(self.processed_dir / "feature_selection.json")["selected"]
            train = table.iloc[split["train_idx"]].reset_index(drop=True)
            rng = np.random.default_rng(self.seed)
            bg = train[selected].iloc[rng.choice(len(train), size=min(N_BACKGROUND, len(train)), replace=False)]
            self._ctx = {"selected": selected, "train": train, "background": bg.reset_index(drop=True)}
            return self._ctx

    @property
    def selected_features(self) -> list[str]:
        return list(self._context()["selected"])

    def available_models(self) -> list[str]:
        return [m for m in PREDICT_MODELS if (self.models_dir / f"{m}.joblib").exists()]

    def _check_name(self, name: str) -> None:
        if name not in PREDICT_MODELS:
            raise ServiceError("MODEL_TIDAK_DIKENAL", f"Model '{name}' tidak dikenal.", 400,
                               {"pilihan": list(PREDICT_MODELS)})
        if name not in self.available_models():
            raise ServiceError("MODEL_BELUM_SIAP", f"File model '{name}' belum ada. Jalankan scripts.run_modeling.", 503)

    def _model(self, name: str):
        self._check_name(name)
        with self._lock:
            if name not in self._models:
                obj, _ = load_model_bundle(name, self.models_dir)
                self._models[name] = obj
            return self._models[name]

    # ------------------------------------------------------------- prediksi
    def raw_scores(self, name: str, X: pd.DataFrame) -> np.ndarray:
        """Skor mentah model (P(fraud) sebelum kalibrasi) untuk fitur terpilih ``X`` (NaN diimputasi oleh model)."""
        model = self._model(name)
        cols = self.selected_features
        if name == "logit_mle":
            return model["model"].predict_proba(model["preprocessor"].transform(X[cols]))[:, 1]
        return np.asarray(model.predict_proba(X[cols]))[:, 1]

    def calibrator(self, name: str) -> PlattCalibrator:
        self._check_name(name)
        with self._lock:
            if name not in self._calibrators:
                ctx = self._context()
                train, y = ctx["train"], ctx["train"][TARGET_COLUMN].to_numpy(int)
                oof_path = self.processed_dir / "oof_predictions.csv"
                if name != "logit_mle" and oof_path.exists():
                    oof = pd.read_csv(oof_path)
                    scores = oof[name].to_numpy(float)       # prediksi out-of-fold: kalibrasi tidak bias terhadap data latih
                else:
                    scores = self.raw_scores(name, train[ctx["selected"]])
                self._calibrators[name] = PlattCalibrator().fit(scores, y)
            return self._calibrators[name]

    def calibrated_probability(self, name: str, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        raw = self.raw_scores(name, X)
        return raw, self.calibrator(name).predict(raw)

    # ------------------------------------------------------------- penjelasan
    def _preprocess(self, name: str, X: pd.DataFrame) -> np.ndarray:
        model = self._model(name)
        pre = model["preprocessor"] if name == "logit_mle" else model.named_steps["pre"]
        return pre.transform(X[self.selected_features]).to_numpy(dtype=float)

    def _explainer(self, name: str):
        with self._lock:
            if name in self._explainers:
                return self._explainers[name]
            ctx = self._context()
            model = self._model(name)
            Zbg = self._preprocess(name, ctx["background"])
            if name == "logit_mle":
                coef, intercept = model["model"].params_.drop("const").to_numpy(), float(model["model"].params_["const"])
                kind, payload = "linear", (coef, intercept)
            elif name == "logreg":
                est = model.named_steps["clf"].estimator_
                kind, payload = "linear", (np.asarray(est.coef_).ravel(), float(np.asarray(est.intercept_).ravel()[0]))
            else:
                kind, payload = "exact", ExactShapleyExplainer(model_scorer(model.named_steps["clf"]), Zbg, ctx["selected"])
            self._explainers[name] = (kind, payload, Zbg)
            return self._explainers[name]

    def explain(self, name: str, X: pd.DataFrame) -> tuple[np.ndarray, float, str]:
        """Atribusi Shapley eksak per fitur: (phi[n, d], nilai_dasar, satuan). Linear: log-odds; pohon: probabilitas."""
        kind, payload, Zbg = self._explainer(name)
        Z = self._preprocess(name, X)
        if kind == "linear":
            coef, intercept = payload
            phi, offset = linear_shap(coef, Z, Zbg)
            return phi, intercept + offset, "log-odds"
        return payload.shap_values(Z), float(payload.expected_value), "probabilitas"

    # ------------------------------------------------------------- informasi
    def default_threshold(self) -> float:
        """Ambang awal: ambang Bayes dari rasio biaya Tahap 5 bila ada, selain itu 1/11 (FN:FP = 10:1)."""
        path = self.processed_dir / "evaluation_results.json"
        if path.exists():
            try:
                return float(_read_json(path)["biaya"]["ambang_bayes"])
            except (KeyError, ValueError, TypeError):
                pass
        return bayes_threshold(10.0, 1.0)

    def feature_ranges(self) -> list[dict[str, Any]]:
        """Rentang nilai fitur pada data latih (untuk slider what-if) + metadata katalog."""
        ctx = self._context()
        cat = {f.name: f for f in FEATURE_CATALOG}
        out = []
        for f in ctx["selected"]:
            x = ctx["train"][f].dropna().to_numpy(dtype=float)
            spec = cat.get(f)
            q = np.quantile(x, [0.01, 0.5, 0.99])
            out.append({"fitur": f, "label": spec.label if spec else f, "rumus": spec.formula if spec else "",
                        "kelompok": spec.group if spec else "", "min": float(x.min()), "p1": float(q[0]), "median": float(q[1]),
                        "p99": float(q[2]), "max": float(x.max())})
        return out

    def model_info(self, name: str) -> dict[str, Any]:
        self._check_name(name)
        meta_path = self.models_dir / f"{name}.json"
        meta = _read_json(meta_path) if meta_path.exists() else {}
        info: dict[str, Any] = {"nama": name, **PREDICT_MODELS[name], "metadata": meta}
        res_path = self.processed_dir / "modeling_results.json"
        if res_path.exists():
            res = _read_json(res_path)
            hold = {r["model"]: r for r in res.get("uji_holdout", [])}
            if name in hold:
                info["metrik_uji"] = hold[name]
            elif name == "logit_mle" and "mle" in res.get("logit_inferensial", {}):
                info["metrik_uji"] = res["logit_inferensial"]["mle"].get("uji")
                info["kecocokan_model"] = res["logit_inferensial"]["mle"].get("kecocokan_model")
            if name in res.get("nested_cv", {}):
                info["nested_cv"] = res["nested_cv"][name]
        return info
