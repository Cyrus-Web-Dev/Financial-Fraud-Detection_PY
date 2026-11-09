"""Katalog model dan pabrik pipeline.

Semua model dibungkus dalam ``Pipeline([("pre", FeaturePreprocessor), ("clf", ResamplingClassifier)])``
sehingga SELURUH langkah yang belajar dari data (winsorize, imputasi, standardisasi, resampling,
pelatihan) hanya melihat data latih pada tiap fold validasi silang.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.features.transform import FeaturePreprocessor
from src.models.compat import make_logreg
from src.models.imbalance import ResamplingClassifier


class IsolationForestScorer(BaseEstimator, ClassifierMixin):
    """Isolation Forest (tak terawasi) dengan antarmuka classifier.

    Skor anomali diubah menjadi nilai 0-1 lewat ECDF skor data latih (peringkat persentil).
    Nilai ini BUKAN probabilitas terkalibrasi; hanya valid untuk metrik peringkat (ROC/PR-AUC, Recall@k).

    Parameters
    ----------
    train_on_normal_only : jika True (semi-terawasi), hutan dilatih hanya pada observasi non-fraud
        sehingga fraud menjadi "anomali" terhadap perilaku normal. Jika False, label diabaikan.
    """

    def __init__(self, n_estimators: int = 300, max_samples: Any = "auto", train_on_normal_only: bool = True,
                 random_state: int = 42) -> None:
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.train_on_normal_only = train_on_normal_only
        self.random_state = random_state

    def fit(self, X, y=None):
        Xa = np.asarray(X, dtype=float)
        if self.train_on_normal_only and y is not None:
            ya = np.asarray(y).astype(int)
            if (ya == 0).sum() >= 10:
                Xa = Xa[ya == 0]
        self.model_ = IsolationForest(n_estimators=self.n_estimators, max_samples=self.max_samples,
                                      random_state=self.random_state).fit(Xa)
        self.train_scores_ = np.sort(-self.model_.score_samples(Xa))
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("Model belum di-fit.")
        return -self.model_.score_samples(np.asarray(X, dtype=float))  # makin besar = makin anomali

    def predict_proba(self, X) -> np.ndarray:
        s = self.decision_function(X)
        p = np.searchsorted(self.train_scores_, s, side="right") / len(self.train_scores_)
        return np.column_stack([1.0 - p, p])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.95).astype(int)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    label: str
    kind: str                                   # "klasik" | "ml" | "anomali"
    factory: Callable[[dict, int], Any]
    default_params: dict[str, Any]
    grid: dict[str, list]
    supports_resampling: bool = True
    interpretable: bool = False
    notes: str = ""


def _logreg(p: dict, seed: int):
    return make_logreg("l2", C=p["C"], random_state=seed)


def _rf(p: dict, seed: int):
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=p["min_samples_leaf"],
                                  max_features=p["max_features"], n_jobs=-1, random_state=seed)


def _hgb(p: dict, seed: int):
    return HistGradientBoostingClassifier(learning_rate=p["learning_rate"], max_leaf_nodes=p["max_leaf_nodes"],
                                          max_iter=150, l2_regularization=p["l2_regularization"],
                                          early_stopping=False, random_state=seed)


def _iforest(p: dict, seed: int):
    return IsolationForestScorer(n_estimators=300, max_samples=p["max_samples"], random_state=seed)


MODEL_SPECS: dict[str, ModelSpec] = {
    "logreg": ModelSpec("logreg", "Regresi Logistik (L2)", "klasik", _logreg, {"C": 1.0},
                        {"C": [0.01, 0.1, 1.0, 10.0]}, interpretable=True,
                        notes="Model parametrik; koefisien dan odds ratio dapat diinterpretasi."),
    "random_forest": ModelSpec("random_forest", "Random Forest", "ml", _rf,
                               {"min_samples_leaf": 5, "max_features": "sqrt"},
                               {"min_samples_leaf": [2, 5, 10], "max_features": ["sqrt", 0.5]},
                               notes="Ensambel pohon; menangkap nonlinearitas dan interaksi."),
    "hist_gb": ModelSpec("hist_gb", "Gradient Boosting", "ml", _hgb,
                         {"learning_rate": 0.05, "max_leaf_nodes": 8, "l2_regularization": 1.0},
                         {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [8, 15], "l2_regularization": [0.0, 1.0]},
                         notes="Boosting berbasis histogram; biasanya akurat pada data tabular."),
    "isolation_forest": ModelSpec("isolation_forest", "Isolation Forest (anomali)", "anomali", _iforest,
                                  {"max_samples": "auto"}, {"max_samples": ["auto", 0.5]},
                                  supports_resampling=False,
                                  notes="Tak terawasi (dilatih pada non-fraud); pembanding berbasis deteksi anomali."),
}


def param_combinations(grid: dict[str, list]) -> list[dict[str, Any]]:
    """Semua kombinasi parameter dari grid (urutan deterministik)."""
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def make_model(name: str, params: dict[str, Any] | None = None, strategy: str = "none", seed: int = 42,
               sampling_ratio: float = 1.0) -> Pipeline:
    """Bangun pipeline: preprocessing -> (resampling +) model. Isolation Forest mengabaikan strategi."""
    if name not in MODEL_SPECS:
        raise ValueError(f"Model '{name}' tidak dikenal. Pilihan: {', '.join(MODEL_SPECS)}")
    spec = MODEL_SPECS[name]
    merged = {**spec.default_params, **(params or {})}
    base = spec.factory(merged, seed)
    if spec.supports_resampling:
        clf = ResamplingClassifier(base, strategy=strategy, sampling_ratio=sampling_ratio, random_state=seed)
    else:
        clf = base
    return Pipeline([("pre", FeaturePreprocessor()), ("clf", clf)])


def catalog_models() -> list[dict]:
    return [{"name": s.name, "label": s.label, "kind": s.kind, "interpretable": s.interpretable,
             "notes": s.notes, "default_params": s.default_params} for s in MODEL_SPECS.values()]
