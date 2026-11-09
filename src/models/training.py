"""Pelatihan dan validasi model.

Prinsip metodologis
-------------------
* Semua validasi silang memakai lipatan stratified-GROUP (per perusahaan), sehingga baris
  perusahaan yang sama tidak pernah berada di latih dan validasi sekaligus.
* Seluruh langkah yang belajar dari data berada di dalam ``Pipeline`` (lihat ``zoo.make_model``).
* Tuning hiperparameter dievaluasi dengan **nested CV**: lipatan luar untuk estimasi kinerja
  yang tidak bias, lipatan dalam untuk memilih hiperparameter. Memilih hiperparameter dan
  melaporkan skor dari lipatan yang sama membuat skor terlalu optimistis.
* Metrik utama: PR-AUC (average precision); pendamping: ROC-AUC, Recall@10%, Precision@10%.

Keterbatasan yang perlu dilaporkan: seleksi fitur (tahap sebelumnya) dilakukan pada seluruh data
latih sebelum CV, sehingga skor CV sedikit optimistis. Kinerja pada data uji yang tidak pernah
disentuh adalah angka akhir yang bersih.
"""
from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn

from src.data.split import group_stratified_cv
from src.evaluation.metrics import ranking_metrics
from src.models.imbalance import STRATEGIES, class_counts, resample
from src.models.zoo import MODEL_SPECS, make_model, param_combinations
from src.utils import to_jsonable

METRIC_COLUMNS: list[str] = ["roc_auc", "pr_auc", "recall_at_10", "precision_at_10"]


@dataclass
class CVResult:
    model: str
    strategy: str
    params: dict[str, Any]
    fold_metrics: pd.DataFrame       # satu baris per lipatan
    oof_scores: np.ndarray           # skor out-of-fold, urutan sama dengan data masukan

    def summary(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for m in METRIC_COLUMNS:
            out[f"{m}_mean"] = float(self.fold_metrics[m].mean())
            out[f"{m}_sd"] = float(self.fold_metrics[m].std(ddof=1)) if len(self.fold_metrics) > 1 else np.nan
        return out


def _as_arrays(X: pd.DataFrame, y, groups) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X harus berupa pandas DataFrame.")
    ya, ga = np.asarray(y).astype(int), np.asarray(groups)
    if not (len(X) == len(ya) == len(ga)):
        raise ValueError("X, y, dan groups harus sama panjang.")
    return X.reset_index(drop=True), ya, ga


def _scores(pipe, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(pipe.predict_proba(X))[:, 1]


def cross_validate_model(X: pd.DataFrame, y, groups, name: str, strategy: str = "none",
                         params: dict[str, Any] | None = None, n_splits: int = 5, seed: int = 42,
                         sampling_ratio: float = 1.0) -> CVResult:
    """Validasi silang stratified-group untuk satu model dan satu strategi imbalance."""
    X, ya, ga = _as_arrays(X, y, groups)
    oof = np.full(len(ya), np.nan)
    rows = []
    for fold, (tr, te) in enumerate(group_stratified_cv(ya, ga, n_splits=n_splits, seed=seed), start=1):
        pipe = make_model(name, params, strategy, seed, sampling_ratio).fit(X.iloc[tr], ya[tr])
        s = _scores(pipe, X.iloc[te])
        oof[te] = s
        rows.append({"lipatan": fold, "n_uji": len(te), "fraud_uji": int(ya[te].sum()),
                     **ranking_metrics(ya[te], s)})
    return CVResult(name, strategy, {**MODEL_SPECS[name].default_params, **(params or {})},
                    pd.DataFrame(rows), oof)


def tune_model(X: pd.DataFrame, y, groups, name: str, strategy: str = "none",
               grid: dict[str, list] | None = None, n_splits: int = 3, seed: int = 42,
               scoring: str = "pr_auc") -> tuple[dict[str, Any], pd.DataFrame]:
    """Grid search dengan CV stratified-group. Mengembalikan (parameter terbaik, tabel semua kombinasi)."""
    X, ya, ga = _as_arrays(X, y, groups)
    combos = param_combinations(grid if grid is not None else MODEL_SPECS[name].grid)
    folds = list(group_stratified_cv(ya, ga, n_splits=n_splits, seed=seed))
    rows = []
    for params in combos:
        vals = []
        for tr, te in folds:
            pipe = make_model(name, params, strategy, seed).fit(X.iloc[tr], ya[tr])
            vals.append(ranking_metrics(ya[te], _scores(pipe, X.iloc[te]))[scoring])
        rows.append({**{f"p_{k}": v for k, v in params.items()}, "skor_mean": float(np.mean(vals)),
                     "skor_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan})
    table = pd.DataFrame(rows)
    best = int(table["skor_mean"].idxmax())
    return combos[best], table


def nested_cv(X: pd.DataFrame, y, groups, name: str, strategy: str = "none",
              grid: dict[str, list] | None = None, n_outer: int = 5, n_inner: int = 3, seed: int = 42,
              scoring: str = "pr_auc") -> pd.DataFrame:
    """Nested CV: estimasi kinerja yang tidak bias bila hiperparameter ikut dipilih dari data.

    Pada tiap lipatan luar, hiperparameter dipilih HANYA dari data latih lipatan itu (lipatan dalam),
    lalu model dilatih ulang pada seluruh data latih lipatan dan diuji pada lipatan luar.
    """
    X, ya, ga = _as_arrays(X, y, groups)
    rows = []
    for fold, (tr, te) in enumerate(group_stratified_cv(ya, ga, n_splits=n_outer, seed=seed), start=1):
        best, _ = tune_model(X.iloc[tr], ya[tr], ga[tr], name, strategy, grid, n_inner, seed + fold, scoring)
        pipe = make_model(name, best, strategy, seed).fit(X.iloc[tr], ya[tr])
        rows.append({"lipatan_luar": fold, "parameter_terpilih": json.dumps(to_jsonable(best), sort_keys=True),
                     **ranking_metrics(ya[te], _scores(pipe, X.iloc[te]))})
    return pd.DataFrame(rows)


def class_distribution_after(y, strategies: Sequence[str] = STRATEGIES, ratio: float = 1.0,
                             seed: int = 42) -> pd.DataFrame:
    """Jumlah kelas pada data latih sebelum dan sesudah tiap strategi (untuk ditampilkan di skripsi)."""
    ya = np.asarray(y).astype(int)
    dummy = np.random.default_rng(seed).normal(size=(len(ya), 2))
    rows = []
    for s in strategies:
        _, yr = resample(dummy, ya, s, ratio=ratio, seed=seed)
        before, after = class_counts(ya), class_counts(yr)
        note = "bobot kelas 'balanced' pada loss (data tidak berubah)" if s == "class_weight" else ""
        rows.append({"strategi": s, "non_fraud_sebelum": before["non_fraud"], "fraud_sebelum": before["fraud"],
                     "non_fraud_sesudah": after["non_fraud"], "fraud_sesudah": after["fraud"],
                     "proporsi_fraud_sesudah": after["fraud"] / (after["fraud"] + after["non_fraud"]),
                     "catatan": note})
    return pd.DataFrame(rows)


def compare_imbalance_strategies(X: pd.DataFrame, y, groups, models: Sequence[str] = ("logreg", "random_forest"),
                                 strategies: Sequence[str] = STRATEGIES, n_splits: int = 5, seed: int = 42,
                                 sampling_ratio: float = 1.0) -> pd.DataFrame:
    """Bandingkan strategi imbalance (sebelum vs sesudah) dengan CV yang sama untuk tiap model."""
    rows = []
    for m in models:
        for s in strategies:
            if s != "none" and not MODEL_SPECS[m].supports_resampling:
                continue
            res = cross_validate_model(X, y, groups, m, s, None, n_splits, seed, sampling_ratio)
            rows.append({"model": m, "strategi": s, **res.summary()})
    return pd.DataFrame(rows)


def evaluate_on_holdout(X_train: pd.DataFrame, y_train, X_test: pd.DataFrame, y_test, name: str,
                        strategy: str = "none", params: dict[str, Any] | None = None, seed: int = 42,
                        sampling_ratio: float = 1.0) -> tuple[Any, dict[str, float], np.ndarray]:
    """Latih pada data latih penuh dan evaluasi pada data uji yang tidak pernah dipakai sebelumnya."""
    ytr, yte = np.asarray(y_train).astype(int), np.asarray(y_test).astype(int)
    pipe = make_model(name, params, strategy, seed, sampling_ratio).fit(X_train.reset_index(drop=True), ytr)
    scores = _scores(pipe, X_test.reset_index(drop=True))
    return pipe, ranking_metrics(yte, scores), scores


# ---------------------------------------------------------------------------
# Penyimpanan model + metadata (reproducible research)
# ---------------------------------------------------------------------------
def library_versions() -> dict[str, str]:
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "scikit-learn": sklearn.__version__}


def save_model_bundle(model: Any, name: str, directory: Path | str, metadata: dict[str, Any]) -> tuple[Path, Path]:
    """Simpan model (``.joblib``) dan metadata (``.json``: tanggal, versi library, parameter, metrik, fitur)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model_path, meta_path = directory / f"{name}.joblib", directory / f"{name}.json"
    meta = {"nama": name, "disimpan_pada": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "versi_library": library_versions(), **metadata}
    joblib.dump(model, model_path, compress=3)   # kompresi: model berbasis pohon bisa belasan MB
    meta_path.write_text(json.dumps(to_jsonable(meta), ensure_ascii=False, indent=2), encoding="utf-8")
    return model_path, meta_path


def load_model_bundle(name: str, directory: Path | str) -> tuple[Any, dict[str, Any]]:
    """Muat model dan metadata. Peringatan: hanya muat file .joblib dari sumber tepercaya (format pickle)."""
    directory = Path(directory)
    model_path, meta_path = directory / f"{name}.joblib", directory / f"{name}.json"
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Bundle model '{name}' tidak lengkap di {directory}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    current = library_versions()
    saved = meta.get("versi_library", {})
    meta["peringatan_versi"] = [f"{k}: disimpan {saved[k]}, terpasang {current[k]}"
                                for k in ("scikit-learn", "numpy", "pandas") if k in saved and saved[k] != current[k]]
    return joblib.load(model_path), meta
