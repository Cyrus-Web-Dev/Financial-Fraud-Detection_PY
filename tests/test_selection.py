"""Uji split, VIF, dan seleksi fitur. Jalankan: python -m tests.test_selection"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import SAMPLE_CLEAN_FILE
from src.data.split import group_stratified_cv, group_stratified_split, split_summary, temporal_split
from src.features.ratios import BENEISH_INDEX_NAMES, compute_ratios
from src.features.selection import (
    choose_l1_c, compute_vif, correlation_filter, iterative_vif_pruning, run_feature_selection,
    stability_selection, univariate_screening,
)
from src.models.compat import make_logreg, sklearn_version


def _sample() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_CLEAN_FILE)


def _panel_signal(n_groups: int = 500, per_group: int = 3, seed: int = 0):
    """Data sintetis: y hanya bergantung pada f0 dan f1; f2..f7 adalah noise."""
    rng = np.random.default_rng(seed)
    n = n_groups * per_group
    groups = np.repeat(np.arange(n_groups), per_group)
    X = pd.DataFrame(rng.normal(size=(n, 8)), columns=[f"f{i}" for i in range(8)])
    logit = -2.4 + 1.6 * X["f0"] + 1.1 * X["f1"]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y, groups


# --------------------------------- split ----------------------------------
def test_group_split_has_no_company_overlap_and_keeps_prevalence() -> None:
    df = _sample()
    tr, te = group_stratified_split(df, 0.2, seed=1)
    s = split_summary(df, tr, te)
    assert s["perusahaan_beririsan"] == 0
    assert s["n_latih"] + s["n_uji"] == len(df) and len(set(tr) & set(te)) == 0
    assert abs(s["proporsi_fraud_latih"] - s["proporsi_fraud_uji"]) < 0.02
    tr2, te2 = group_stratified_split(df, 0.2, seed=1)
    assert np.array_equal(tr, tr2) and np.array_equal(te, te2)     # reproducible
    assert not np.array_equal(te, group_stratified_split(df, 0.2, seed=2)[1])


def test_group_cv_folds_do_not_split_companies() -> None:
    df = _sample()
    seen = np.zeros(len(df), dtype=int)
    for tr, te in group_stratified_cv(df["is_fraud"], df["company_id"], n_splits=5, seed=0):
        assert not (set(df["company_id"].iloc[tr]) & set(df["company_id"].iloc[te]))
        seen[te] += 1
    assert (seen == 1).all()                                       # tiap baris diuji tepat satu kali


def test_temporal_split() -> None:
    df = _sample()
    tr, te = temporal_split(df, 2)
    assert df["tahun"].iloc[tr].max() < df["tahun"].iloc[te].min()
    assert sorted(df["tahun"].iloc[te].unique()) == [2022, 2023]
    for bad in (0, 6):
        try:
            temporal_split(df, bad)
        except ValueError:
            continue
        raise AssertionError("n_test_years tidak valid harus ditolak")


def test_split_argument_validation() -> None:
    df = _sample()
    try:
        group_stratified_split(df, 0.9)
    except ValueError:
        pass
    else:
        raise AssertionError
    y = np.r_[np.ones(4), np.zeros(96)]; g = np.arange(100)
    try:
        list(group_stratified_cv(y, g, n_splits=5))
    except ValueError as e:
        assert "Kelas 1" in str(e)
    else:
        raise AssertionError("kelas dengan terlalu sedikit grup harus ditolak")


# ---------------------------------- VIF -----------------------------------
def test_vif_matches_inverse_correlation_and_closed_form() -> None:
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=2000); x2 = 0.9 * x1 + rng.normal(scale=0.5, size=2000)
    X = pd.DataFrame({"x1": x1, "x2": x2})
    r = np.corrcoef(x1, x2)[0, 1]
    vif = compute_vif(X)
    assert np.allclose(vif, 1 / (1 - r ** 2))                      # bentuk tertutup dua variabel
    X3 = X.assign(x3=rng.normal(size=2000) + 0.3 * x1)
    inv_diag = np.diag(np.linalg.inv(np.corrcoef(X3.to_numpy().T)))
    assert np.allclose(compute_vif(X3).to_numpy(), inv_diag)       # identik dengan diag(R^-1)


def test_vif_independent_and_perfect_collinearity_and_errors() -> None:
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(5000, 4)), columns=list("abcd"))
    assert (compute_vif(X) < 1.05).all()
    X["e"] = X["a"] + X["b"]
    v = compute_vif(X)
    assert v[["a", "b", "e"]].min() > 1e5                          # kolinearitas sempurna -> sangat besar
    for bad in (X.iloc[:, :1], X.assign(a=np.nan)):
        try:
            compute_vif(bad)
        except ValueError:
            continue
        raise AssertionError("harus ValueError")


def test_iterative_vif_pruning_and_protection() -> None:
    rng = np.random.default_rng(2)
    a = rng.normal(size=1500)
    X = pd.DataFrame({"a": a, "b": a + rng.normal(scale=0.05, size=1500), "c": rng.normal(size=1500),
                      "d": rng.normal(size=1500)})
    kept, log, final = iterative_vif_pruning(X, 10.0)
    assert len(kept) == 3 and "c" in kept and "d" in kept and (final <= 10).all()
    assert len(log) == 1 and log.loc[0, "vif"] > 10
    kept, log, final = iterative_vif_pruning(X, 10.0, protected=["a", "b"])
    assert {"a", "b"} <= set(kept) and log.empty                   # terlindungi tidak dibuang
    assert (final[["a", "b"]] > 10).all()                          # dan kondisi itu terlihat di VIF akhir


def test_correlation_filter_drops_weaker_and_respects_protection() -> None:
    rng = np.random.default_rng(3)
    a = rng.normal(size=500)
    X = pd.DataFrame({"a": a, "b": a * 2 + rng.normal(scale=0.01, size=500), "c": rng.normal(size=500)})
    strength = pd.Series({"a": 0.7, "b": 0.6, "c": 0.5})
    kept, drops = correlation_filter(X, strength, 0.9)
    assert kept == ["a", "c"] and drops.loc[0, "fitur"] == "b" and drops.loc[0, "berkorelasi_dengan"] == "a"
    kept, drops = correlation_filter(X, strength, 0.9, protected=["b"])
    assert kept == ["b", "c"]


# ------------------------- skrining dan Lasso -----------------------------
def test_univariate_auc_matches_sklearn() -> None:
    X, y, _ = _panel_signal()
    res = univariate_screening(X, y).set_index("fitur")
    for col in ("f0", "f1", "f5"):
        assert math.isclose(res.loc[col, "auc"], roc_auc_score(y, X[col]), abs_tol=1e-9)
    assert res.loc["f0", "auc"] > 0.7 and res.loc["f0", "arah"] == "+"
    assert res.loc["f0", "q_value"] < 1e-6 and res.loc["f5", "auc_kuat"] < 0.58


def test_l1_helper_really_applies_l1() -> None:
    X, y, _ = _panel_signal()
    l1 = make_logreg("l1", C=0.01).fit(X, y)
    l2 = make_logreg("l2", C=0.01).fit(X, y)
    assert (l1.coef_ == 0).sum() >= 4 and (l2.coef_ == 0).sum() == 0
    assert sklearn_version() >= (1, 0)
    try:
        make_logreg("elasticnet")
    except ValueError:
        pass
    else:
        raise AssertionError


def test_stability_selection_recovers_true_signal() -> None:
    X, y, groups = _panel_signal()
    C, table = choose_l1_c(X, y, groups, seed=0)                   # C dipilih CV (aturan 1-SE), seperti di pipeline
    assert table["C"].min() <= C <= table["C"].max() and len(table) == 10
    freq = stability_selection(X, y, groups, C=C, n_boot=60, seed=0)
    assert freq[["f0", "f1"]].min() > 0.9                          # sinyal sejati selalu terpilih
    assert freq[[f"f{i}" for i in range(2, 8)]].max() < 0.2        # noise hampir tidak pernah terpilih


def test_weak_penalty_admits_noise_but_strong_penalty_does_not() -> None:
    """Dokumentasi perilaku: C terlalu besar (penalti lemah) meloloskan noise."""
    X, y, groups = _panel_signal()
    weak = stability_selection(X, y, groups, C=0.5, n_boot=30, seed=0)
    strong = stability_selection(X, y, groups, C=0.01, n_boot=30, seed=0)
    assert weak.iloc[2:].mean() > strong.iloc[2:].mean() + 0.3


# ------------------------------ pipeline penuh ----------------------------
def test_full_selection_pipeline_on_sample() -> None:
    df = _sample()
    tr, te = group_stratified_split(df, 0.2, seed=42)
    F = compute_ratios(df)
    kw = dict(X_train=F.iloc[tr], y_train=df["is_fraud"].iloc[tr], groups_train=df["company_id"].iloc[tr],
              n_boot=30, seed=42)
    res = run_feature_selection(**kw)
    assert set(BENEISH_INDEX_NAMES) <= set(res.selected)           # fitur domain dipertahankan
    assert {"dsri", "tata", "round_share_1000"} <= set(res.selected)
    assert set(res.selected) <= set(F.columns) and 8 <= len(res.selected) < len(F.columns)
    assert not res.steps["fitur_terlindungi_vif_tinggi"]
    assert (res.steps["vif_akhir"] <= 10).all()
    rep = res.report
    assert len(rep) == len(F.columns) and set(rep["status"]) == {"dipertahankan", "dibuang"}
    assert (rep.loc[rep.status == "dibuang", "alasan"].str.len() > 5).all()
    # preprocessor final bisa dipakai pada data UJI tanpa kebocoran
    Xte = res.preprocessor.transform(F.iloc[te])
    assert list(Xte.columns) == res.selected and not Xte.isna().any().any()
    # deterministik dan dapat diserialisasi
    assert run_feature_selection(**kw).selected == res.selected
    json.dumps(res.to_dict(), allow_nan=False)


def test_selection_input_validation() -> None:
    X, y, groups = _panel_signal(n_groups=100)
    for args in ((X.iloc[:-1], y, groups), (X, np.zeros(len(y), dtype=int), groups)):
        try:
            run_feature_selection(*args, n_boot=5)
        except ValueError:
            continue
        raise AssertionError("harus ValueError")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"OK  {t.__name__}")
    print(f"\n{len(tests)} uji lulus.")
