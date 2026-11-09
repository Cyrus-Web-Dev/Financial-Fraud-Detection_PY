"""Uji metrik, zoo model, CV, nested CV, dan persistensi. Jalankan: python -m tests.test_models"""
from __future__ import annotations

import json
import math
import tempfile

import numpy as np
import pandas as pd

from src.evaluation.metrics import probability_metrics, ranking_metrics, recall_precision_at_k, threshold_metrics
from src.models.training import (
    class_distribution_after, compare_imbalance_strategies, cross_validate_model, evaluate_on_holdout,
    load_model_bundle, nested_cv, save_model_bundle, tune_model,
)
from src.models.zoo import MODEL_SPECS, IsolationForestScorer, catalog_models, make_model, param_combinations


def _signal(n_groups: int = 400, per: int = 3, seed: int = 0):
    """Data dengan sinyal pada f0 dan f1; y ~ 12% fraud."""
    rng = np.random.default_rng(seed)
    n = n_groups * per
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"f{i}" for i in range(5)])
    y = (rng.random(n) < 1 / (1 + np.exp(-(-2.6 + 1.5 * X.f0 + 1.0 * X.f1)))).astype(int)
    return X, y, np.repeat(np.arange(n_groups), per)


# --------------------------------- metrik ---------------------------------
def test_ranking_metrics_hand_computed() -> None:
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    perfect = np.array([.9, .8, .1, .2, .3, .1, .2, .1, .3, .2])
    m = ranking_metrics(y, perfect, k=0.2)
    assert m["roc_auc"] == 1.0 and m["pr_auc"] == 1.0 and m["recall_at_20"] == 1.0 and m["precision_at_20"] == 1.0
    worst = ranking_metrics(y, 1 - perfect, k=0.2)
    assert worst["roc_auc"] == 0.0 and worst["recall_at_20"] == 0.0
    # satu fraud di peringkat 1, satu di peringkat 3 dari 10: recall@20% = 0.5, precision@20% = 0.5
    s = np.array([.9, .5, .8, .1, .1, .1, .1, .1, .1, .1])
    rec, prec = recall_precision_at_k(y, s, 0.2)
    assert rec == 0.5 and prec == 0.5
    assert all(np.isnan(v) for v in ranking_metrics(np.zeros(5), np.arange(5.0)).values())   # satu kelas
    for bad in (lambda: recall_precision_at_k(y, s, 0.0), lambda: ranking_metrics(y, s[:-1]),
                lambda: ranking_metrics(y, np.r_[s[:-1], np.nan])):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("harus ValueError")


def test_threshold_and_probability_metrics() -> None:
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0]); p = np.array([.9, .8, .2, .6, .1, .1, .1, .1, .1, .1])
    m = threshold_metrics(y, p, 0.5)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (2, 1, 6, 1)
    assert math.isclose(m["precision"], 2 / 3) and math.isclose(m["recall"], 2 / 3) and math.isclose(m["specificity"], 6 / 7)
    assert math.isclose(m["f1"], 2 / 3) and math.isclose(m["accuracy"], 0.8)
    assert math.isclose(m["balanced_accuracy"], (2 / 3 + 6 / 7) / 2)
    none_flagged = threshold_metrics(y, np.zeros(10), 0.5)
    assert np.isnan(none_flagged["precision"]) and none_flagged["recall"] == 0.0
    assert math.isclose(probability_metrics(np.array([1, 0]), np.array([1.0, 0.0]))["brier"], 0.0)
    assert math.isclose(probability_metrics(np.array([1, 0]), np.array([0.5, 0.5]))["brier"], 0.25)


# ----------------------------------- zoo -----------------------------------
def test_isolation_forest_scorer() -> None:
    rng = np.random.default_rng(0)
    normal = rng.normal(size=(500, 3)); outliers = rng.normal(6, 1, size=(10, 3))
    X = np.vstack([normal, outliers]); y = np.r_[np.zeros(500), np.ones(10)].astype(int)
    m = IsolationForestScorer(n_estimators=100, random_state=0).fit(X, y)
    assert len(m.train_scores_) == 500                                  # hanya non-fraud untuk pelatihan
    p = m.predict_proba(X)[:, 1]
    assert (p >= 0).all() and (p <= 1).all() and np.allclose(m.predict_proba(X).sum(axis=1), 1)
    assert p[500:].min() > p[:500].mean() and ranking_metrics(y, p)["roc_auc"] > 0.95
    assert len(IsolationForestScorer(train_on_normal_only=False, random_state=0).fit(X, y).train_scores_) == 510
    try:
        IsolationForestScorer().decision_function(X)
    except RuntimeError:
        pass
    else:
        raise AssertionError("harus RuntimeError")


def test_make_model_all_specs_fit_and_predict() -> None:
    X, y, _ = _signal(n_groups=200)
    for name in MODEL_SPECS:
        for strat in (("none", "class_weight", "smote") if MODEL_SPECS[name].supports_resampling else ("none", "smote")):
            pipe = make_model(name, None, strat, 0).fit(X, y)
            p = pipe.predict_proba(X)[:, 1]
            assert p.shape == (len(X),) and np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all(), (name, strat)
    assert [s[0] for s in make_model("logreg").steps] == ["pre", "clf"]
    try:
        make_model("xyz")
    except ValueError:
        pass
    else:
        raise AssertionError("model tidak dikenal harus ditolak")
    assert len(catalog_models()) == 4 and len(param_combinations({"a": [1, 2], "b": [3, 4, 5]})) == 6
    assert param_combinations({"a": [1, 2]}) == [{"a": 1}, {"a": 2}]


def test_pipeline_preprocessing_is_fit_inside_each_fold_only() -> None:
    """Sebuah nilai ekstrem di data uji tidak boleh memengaruhi parameter preprocessing."""
    X, y, _ = _signal(n_groups=200)
    Xtr, Xte = X.iloc[:400].copy(), X.iloc[400:].copy()
    Xte.iloc[0, 0] = 1e9
    pipe = make_model("logreg", None, "none", 0).fit(Xtr, y[:400])
    assert pipe.named_steps["pre"].upper_["f0"] < 10                    # batas winsor dari data latih saja
    assert np.isfinite(pipe.predict_proba(Xte)[:, 1]).all()


# ------------------------------- CV dan tuning -------------------------------
def test_cross_validate_model() -> None:
    X, y, g = _signal()
    r = cross_validate_model(X, y, g, "logreg", "class_weight", n_splits=5, seed=0)
    assert len(r.fold_metrics) == 5 and r.fold_metrics["n_uji"].sum() == len(y)
    assert np.isfinite(r.oof_scores).all() and len(r.oof_scores) == len(y)
    s = r.summary()
    assert s["roc_auc_mean"] > 0.8 and s["pr_auc_mean"] > 0.35 and s["pr_auc_sd"] >= 0
    r2 = cross_validate_model(X, y, g, "logreg", "class_weight", n_splits=5, seed=0)
    assert np.array_equal(r.oof_scores, r2.oof_scores)                  # reproducible
    Xs = X.copy(); Xs.index = np.arange(1000, 1000 + len(X))             # indeks non-standar tidak boleh merusak
    assert np.allclose(cross_validate_model(Xs, y, g, "logreg", "none", n_splits=5, seed=0).oof_scores,
                       cross_validate_model(X, y, g, "logreg", "none", n_splits=5, seed=0).oof_scores)
    for bad in (lambda: cross_validate_model(X.to_numpy(), y, g, "logreg"), lambda: cross_validate_model(X, y[:-1], g, "logreg")):
        try:
            bad()
        except (TypeError, ValueError):
            continue
        raise AssertionError("harus ditolak")


def test_tune_model_and_nested_cv() -> None:
    X, y, g = _signal()
    grid = {"C": [0.0001, 0.1, 1.0]}
    best, table = tune_model(X, y, g, "logreg", "none", grid, n_splits=3, seed=0)
    assert best["C"] in grid["C"] and len(table) == 3 and best["C"] != 0.0001      # regularisasi ekstrem kalah
    assert table["skor_mean"].max() == table.loc[table["p_C"] == best["C"], "skor_mean"].iloc[0]
    res = nested_cv(X, y, g, "logreg", "none", grid, n_outer=4, n_inner=3, seed=0)
    assert len(res) == 4 and res["pr_auc"].between(0, 1).all()
    for js in res["parameter_terpilih"]:
        assert json.loads(js)["C"] in grid["C"]


def test_compare_imbalance_strategies_table() -> None:
    X, y, g = _signal(n_groups=250)
    t = compare_imbalance_strategies(X, y, g, models=("logreg", "isolation_forest"), n_splits=3, seed=0)
    assert len(t[t.model == "logreg"]) == 5 and list(t[t.model == "isolation_forest"]["strategi"]) == ["none"]
    assert {"pr_auc_mean", "roc_auc_sd", "recall_at_10_mean"} <= set(t.columns)
    assert t["roc_auc_mean"].between(0.5, 1.0).all()


def test_evaluate_on_holdout() -> None:
    X, y, g = _signal()
    pipe, m, s = evaluate_on_holdout(X.iloc[:900], y[:900], X.iloc[900:], y[900:], "logreg", "class_weight", seed=0)
    assert m["roc_auc"] > 0.8 and len(s) == 300 and hasattr(pipe, "predict_proba")


# ------------------------------- persistensi -------------------------------
def test_save_and_load_model_bundle_roundtrip() -> None:
    X, y, _ = _signal(n_groups=150)
    pipe = make_model("random_forest", {"min_samples_leaf": 5}, "class_weight", 0).fit(X, y)
    with tempfile.TemporaryDirectory() as d:
        mp, jp = save_model_bundle(pipe, "rf_test", d, {"fitur": list(X.columns), "seed": 0, "metrik": {"pr_auc": 0.5}})
        assert mp.exists() and jp.exists()
        model, meta = load_model_bundle("rf_test", d)
        assert np.array_equal(model.predict_proba(X), pipe.predict_proba(X))
        assert meta["fitur"] == list(X.columns) and meta["versi_library"]["scikit-learn"] and meta["disimpan_pada"]
        assert meta["peringatan_versi"] == []
        json.loads(jp.read_text(encoding="utf-8"))
        try:
            load_model_bundle("tidak_ada", d)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("harus FileNotFoundError")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"OK  {t.__name__}")
    print(f"\n{len(tests)} uji lulus.")
