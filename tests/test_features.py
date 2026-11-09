"""Uji rasio keuangan, Beneish/Altman, dan FeaturePreprocessor. Jalankan: python -m tests.test_features"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.config import SAMPLE_CLEAN_FILE
from src.data.schema import NUMERIC_COLUMNS
from src.features.ratios import (
    BENEISH_INDEX_NAMES, FEATURE_CATALOG, FEATURE_NAMES, beneish_m_score, catalog_as_records,
    compute_benchmark_scores, compute_ratios, safe_divide,
)
from src.features.transform import FeaturePreprocessor

# Laporan keuangan sederhana: semua angka bisa dihitung tangan
ROW = dict(
    total_assets=1000, current_assets=400, cash=100, receivables=150, inventory=100, ppe_net=500,
    current_liabilities=200, long_term_debt=300, total_liabilities=500, equity=500, retained_earnings=250,
    revenue=1200, cogs=840, sga=120, depreciation=50, ebit=190, interest_expense=30, net_income=120,
    operating_cash_flow=100, market_value_equity=1500,
    total_assets_prev=900, current_assets_prev=350, receivables_prev=100, ppe_net_prev=450,
    current_liabilities_prev=180, long_term_debt_prev=220, revenue_prev=1000, cogs_prev=650,
    sga_prev=90, depreciation_prev=45,
)


def _one(**override) -> pd.DataFrame:
    row = {**ROW, **override}
    return pd.DataFrame([row])[NUMERIC_COLUMNS]


def _sample() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_CLEAN_FILE)


# ------------------------------ katalog -----------------------------------
def test_catalog_consistency() -> None:
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)) == len(FEATURE_CATALOG)
    assert set(BENEISH_INDEX_NAMES) <= set(FEATURE_NAMES)
    assert all(f.expected_sign in (-1, 0, 1) for f in FEATURE_CATALOG)
    assert all(f.label and f.formula and f.rationale and f.group for f in FEATURE_CATALOG)
    assert list(compute_ratios(_sample().head(50)).columns) == FEATURE_NAMES
    assert len(catalog_as_records()) == len(FEATURE_CATALOG)


def test_safe_divide() -> None:
    out = safe_divide(pd.Series([1.0, 2.0, 3.0, -1.0]), pd.Series([2.0, 0.0, np.nan, 1e-12]))
    assert out.iloc[0] == 0.5 and out.iloc[1:].isna().all()
    assert not np.isinf(safe_divide(pd.Series([1.0]), pd.Series([0.0]))).any()


# ------------------------- nilai acuan hitung tangan ----------------------
def test_beneish_indices_hand_computed() -> None:
    r = compute_ratios(_one()).iloc[0]
    assert math.isclose(r["dsri"], 1.25, rel_tol=1e-12)
    assert math.isclose(r["gmi"], 0.35 / 0.30, rel_tol=1e-12)
    assert math.isclose(r["aqi"], 0.1 / (1 - 800 / 900), rel_tol=1e-12)
    assert math.isclose(r["sgi"], 1.2, rel_tol=1e-12)
    assert math.isclose(r["depi"], (45 / 495) / (50 / 550), rel_tol=1e-12)
    assert math.isclose(r["sgai"], 0.1 / 0.09, rel_tol=1e-12)
    assert math.isclose(r["lvgi"], 0.5 / (400 / 900), rel_tol=1e-12)
    assert math.isclose(r["tata"], 0.02, rel_tol=1e-12)


def test_m_score_hand_computed() -> None:
    expected = (-4.84 + 0.920 * 1.25 + 0.528 * (0.35 / 0.30) + 0.404 * 0.9 + 0.892 * 1.2
                + 0.115 * 1.0 - 0.172 * (0.1 / 0.09) + 4.679 * 0.02 - 0.327 * 1.125)
    assert math.isclose(expected, -1.990406, abs_tol=1e-5)  # nilai yang dihitung tangan
    r = compute_ratios(_one())
    assert math.isclose(beneish_m_score(r).iloc[0], expected, abs_tol=1e-9)
    b = compute_benchmark_scores(_one())
    assert math.isclose(b["m_score"].iloc[0], expected, abs_tol=1e-9)
    assert b["m_flag"].iloc[0] == 0.0                       # -1.99 < -1.78
    flagged = compute_benchmark_scores(_one(receivables=400, net_income=300, operating_cash_flow=10))
    assert flagged["m_flag"].iloc[0] == 1.0


def test_altman_hand_computed() -> None:
    b = compute_benchmark_scores(_one()).iloc[0]
    assert math.isclose(b["altman_z"], 1.2 * 0.2 + 1.4 * 0.25 + 3.3 * 0.19 + 0.6 * 3.0 + 1.0 * 1.2, rel_tol=1e-12)
    assert math.isclose(b["altman_z"], 4.217, abs_tol=1e-9) and b["altman_z_zone"] == "Aman"
    assert math.isclose(b["altman_z2"], 3.25 + 6.56 * 0.2 + 3.26 * 0.25 + 6.72 * 0.19 + 1.05 * 1.0, rel_tol=1e-12)
    # Z = 1.2(-0.1) + 1.4(-0.4) + 3.3(-0.1) + 0.6(0.2) + 1.0(0.6) = -0.29 -> Distress
    distress = compute_benchmark_scores(_one(retained_earnings=-400, ebit=-100, current_assets=100,
                                             market_value_equity=100, revenue=600)).iloc[0]
    assert math.isclose(distress["altman_z"], -0.29, abs_tol=1e-9)
    assert distress["altman_z_zone"] == "Distress"


def test_other_ratios_hand_computed() -> None:
    r = compute_ratios(_one()).iloc[0]
    exp = dict(current_ratio=2.0, quick_ratio=1.5, cash_ratio=0.5, debt_to_assets=0.5, debt_to_equity=1.0,
               long_term_debt_to_assets=0.3, interest_coverage=190 / 30, roa=0.12, roe=0.24, net_margin=0.1,
               gross_margin=0.3, operating_margin=190 / 1200, ebit_to_assets=0.19, asset_turnover=1.2,
               dso=150 / 1200 * 365, dio=100 / 840 * 365, receivables_to_assets=0.15, inventory_to_assets=0.1,
               ppe_to_assets=0.5, retained_earnings_to_assets=0.25, working_capital_to_assets=0.2,
               asset_growth=1000 / 900 - 1, cfo_to_assets=0.1, cfo_to_revenue=100 / 1200,
               mve_to_liabilities=3.0, market_to_book=3.0, log_total_assets=math.log(1000),
               round_share_1000=1 / 3)
    for k, v in exp.items():
        assert math.isclose(r[k], v, rel_tol=1e-9, abs_tol=1e-12), (k, r[k], v)
    assert set(exp) | set(BENEISH_INDEX_NAMES) == set(FEATURE_NAMES)  # tidak ada fitur yang terlewat


# ------------------------------ kasus tepi --------------------------------
def test_negative_equity_makes_equity_ratios_nan() -> None:
    r = compute_ratios(_one(equity=-50)).iloc[0]
    assert np.isnan(r["roe"]) and np.isnan(r["debt_to_equity"]) and np.isnan(r["market_to_book"])
    assert not np.isnan(r["roa"])


def test_zero_denominators_give_nan_not_inf() -> None:
    r = compute_ratios(_one(revenue=0, interest_expense=0, cogs=0)).iloc[0]
    assert np.isnan(r["interest_coverage"]) and np.isnan(r["net_margin"]) and np.isnan(r["dso"])
    assert not np.isinf(r.to_numpy(dtype=float)).any()


def test_missing_column_and_index_preserved() -> None:
    try:
        compute_ratios(_one().drop(columns=["revenue"]))
    except ValueError as e:
        assert "revenue" in str(e)
    else:
        raise AssertionError("harus gagal bila kolom hilang")
    df = _sample().head(20); df.index = np.arange(100, 120)
    assert compute_ratios(df).index.equals(df.index)


def test_ratios_on_sample_are_finite_and_input_not_mutated() -> None:
    df = _sample(); snap = df.copy(deep=True)
    f = compute_ratios(df)
    pd.testing.assert_frame_equal(df, snap)
    assert not np.isinf(f.to_numpy()).any()
    assert f.isna().mean().max() < 0.01
    b = compute_benchmark_scores(df)
    assert b.loc[df.is_fraud == 1, "m_flag"].mean() > 5 * b.loc[df.is_fraud == 0, "m_flag"].mean()


# --------------------------- FeaturePreprocessor --------------------------
def test_preprocessor_standardizes_train_and_uses_train_params_on_test() -> None:
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"a": rng.normal(10, 2, 500), "b": rng.lognormal(size=500)})
    test = pd.DataFrame({"a": [10.0, 1000.0], "b": [1.0, -5.0]})
    pre = FeaturePreprocessor().fit(train)
    t = pre.transform(train)
    assert np.allclose(t.mean(), 0, atol=1e-9) and np.allclose(t.std(ddof=0), 1, atol=1e-9)
    before = (dict(pre.mean_), dict(pre.upper_))
    out = pre.transform(test)
    assert (dict(pre.mean_), dict(pre.upper_)) == before          # transform tidak mengubah parameter
    expected_a = (pre.upper_["a"] - pre.mean_["a"]) / pre.std_["a"]
    assert math.isclose(out.loc[1, "a"], expected_a)               # 1000 di-winsorize ke batas latih
    assert out.loc[1, "b"] < out.loc[0, "b"]


def test_preprocessor_imputes_with_train_median_and_handles_inf() -> None:
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan]})
    pre = FeaturePreprocessor(scale=False, lower_q=0.0, upper_q=1.0).fit(train)
    out = pre.transform(pd.DataFrame({"a": [np.nan, np.inf, -np.inf, 2.0]}))
    assert out["a"].tolist() == [3.0, 3.0, 3.0, 2.0]               # median latih = 3


def test_preprocessor_drops_bad_columns_with_reasons() -> None:
    rng = np.random.default_rng(1)
    train = pd.DataFrame({"ok": rng.normal(size=100), "const": 5.0, "empty": np.nan,
                          "mostly_nan": np.where(rng.random(100) < 0.6, np.nan, 1.0) * rng.normal(size=100)})
    pre = FeaturePreprocessor(max_missing_share=0.4).fit(train)
    assert pre.columns_ == ["ok"]
    assert set(pre.dropped_) == {"const", "empty", "mostly_nan"}
    assert "sd = 0" in pre.dropped_["const"] and "kosong" in pre.dropped_["empty"]
    assert pre.transform(train.assign(extra=1)).columns.tolist() == ["ok"]


def test_preprocessor_errors_and_sklearn_api() -> None:
    pre = FeaturePreprocessor()
    for bad in (lambda: pre.transform(pd.DataFrame({"a": [1.0]})), lambda: pre.get_feature_names_out()):
        try:
            bad()
        except RuntimeError:
            continue
        raise AssertionError("harus RuntimeError sebelum fit")
    for kwargs in (dict(lower_q=0.9, upper_q=0.1), ):
        try:
            FeaturePreprocessor(**kwargs).fit(pd.DataFrame({"a": [1.0, 2.0, 3.0]}))
        except ValueError:
            continue
        raise AssertionError("kuantil terbalik harus ditolak")
    fitted = FeaturePreprocessor().fit(pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 1.0, 3.0, 2.0]}))
    try:
        fitted.transform(pd.DataFrame({"a": [1.0]}))
    except ValueError as e:
        assert "b" in str(e)
    else:
        raise AssertionError("kolom hilang harus ditolak")
    c = clone(FeaturePreprocessor(lower_q=0.05, scale=False))
    assert c.lower_q == 0.05 and c.scale is False and not hasattr(c, "columns_")
    assert fitted.get_feature_names_out().tolist() == ["a", "b"]
    assert len(fitted.parameters_table()) == 2


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"OK  {t.__name__}")
    print(f"\n{len(tests)} uji lulus.")
