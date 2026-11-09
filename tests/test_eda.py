"""Uji modul EDA dan deteksi outlier. Jalankan: python -m tests.test_eda"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import SAMPLE_CLEAN_FILE
from src.data.quality import detect_outliers, modified_z_score, outlier_mask
from src.evaluation.eda import (
    benjamini_hochberg, class_distribution, compare_groups, correlation_matrix,
    descriptive_statistics, high_correlation_pairs, normality_tests, wilson_interval,
)
from src.evaluation.eda_summary import build_eda_summary


def _df() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_CLEAN_FILE)


def test_descriptive_statistics_matches_numpy() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    out = descriptive_statistics(pd.DataFrame({"a": x}), ["a"]).iloc[0]
    assert out["n"] == 5 and out["missing"] == 0
    assert abs(out["mean"] - x.mean()) < 1e-12
    assert abs(out["sd"] - x.std(ddof=1)) < 1e-12
    assert out["median"] == 3.0 and out["skewness"] > 1


def test_descriptive_statistics_handles_missing_and_empty() -> None:
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan] * 3})
    out = descriptive_statistics(df).set_index("kolom")
    assert out.loc["a", "missing"] == 1 and out.loc["b", "n"] == 0
    assert np.isnan(out.loc["b", "mean"])


def test_normality_rejects_lognormal_and_keeps_alpha_on_normal() -> None:
    # Data lognormal harus selalu ditolak
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"logn": rng.lognormal(size=500), "const": np.ones(500)})
    out = normality_tests(df).set_index("kolom")
    assert out.loc["logn", "normal"] == "tidak"
    assert out.loc["const", "normal"] == "tidak dapat diuji"
    # Pada data normal, laju penolakan H0 harus mendekati alpha = 0.05 (bukan satu seed saja)
    rejects = []
    for seed in range(200):
        d = pd.DataFrame({"x": np.random.default_rng(seed).normal(size=300)})
        rejects.append(normality_tests(d).loc[0, "normal"] == "tidak")
    assert 0.01 < np.mean(rejects) < 0.12, np.mean(rejects)


def test_correlation_and_pairs() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=300)
    df = pd.DataFrame({"a": a, "b": a * 2 + rng.normal(scale=0.01, size=300), "c": rng.normal(size=300)})
    corr = correlation_matrix(df)
    assert corr.shape == (3, 3) and abs(corr.loc["a", "a"] - 1) < 1e-12
    pairs = high_correlation_pairs(corr, 0.9)
    assert len(pairs) == 1 and {pairs.loc[0, "variabel_1"], pairs.loc[0, "variabel_2"]} == {"a", "b"}
    try:
        correlation_matrix(df, method="xyz")
    except ValueError:
        pass
    else:
        raise AssertionError("method tidak valid harus ditolak")


def test_wilson_interval_reference_values() -> None:
    lo, hi = wilson_interval(0, 100)
    assert abs(lo) < 1e-12 and abs(hi - 0.0370) < 5e-4
    lo, hi = wilson_interval(50, 100)
    assert abs(lo - 0.4038) < 1e-3 and abs(hi - 0.5962) < 1e-3
    assert np.isnan(wilson_interval(1, 0)[0])


def test_benjamini_hochberg_hand_computed() -> None:
    q = benjamini_hochberg([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(q, [0.02, 0.04, 0.04, 0.02])
    q = benjamini_hochberg([0.5, np.nan, 0.01])
    assert np.isnan(q[1]) and abs(q[2] - 0.02) < 1e-12 and abs(q[0] - 0.5) < 1e-12


def test_compare_groups_detects_shift() -> None:
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(60), np.zeros(540)]
    df = pd.DataFrame({"is_fraud": y, "signal": np.r_[rng.normal(1.5, 1, 60), rng.normal(0, 1, 540)],
                       "noise": rng.normal(size=600)})
    out = compare_groups(df).set_index("variabel")
    assert out.loc["signal", "p_value"] < 1e-6 and out.loc["signal", "efek_rank_biserial"] > 0.5
    assert out.loc["noise", "p_value"] > 0.01
    assert out.loc["signal", "q_value"] >= out.loc["signal", "p_value"]


def test_class_distribution_variants() -> None:
    df = _df()
    total = class_distribution(df).iloc[0]
    assert total["n"] == len(df) and total["fraud"] == df["is_fraud"].sum()
    by_year = class_distribution(df, by="tahun")
    assert by_year["n"].sum() == len(df) and len(by_year) == df["tahun"].nunique()
    by_sector = class_distribution(df, by="sektor")
    assert (by_sector["ci95_bawah"] <= by_sector["proporsi_fraud"]).all()
    assert (by_sector["proporsi_fraud"] <= by_sector["ci95_atas"]).all()


def test_modified_z_score_robust() -> None:
    s = pd.Series([10.0, 11, 9, 10, 10, 11, 9, 1000])
    z = modified_z_score(s)
    assert z.iloc[-1] > 50 and z.iloc[:-1].abs().max() < 3
    assert modified_z_score(pd.Series([5.0] * 5)).isna().all()


def test_outliers_clean_vs_dirty() -> None:
    from src.data.synthetic import inject_data_issues
    clean = _df()
    dirty = inject_data_issues(clean, missing_rate=0.0, outlier_rate=0.005, seed=3)
    n_clean = int(outlier_mask(clean).to_numpy().sum())
    n_dirty = int(outlier_mask(dirty).to_numpy().sum())
    assert n_clean <= 10, n_clean            # false positive sangat sedikit pada data bersih
    assert n_dirty > 300, n_dirty            # outlier suntikan (~590 sel) sebagian besar tertangkap
    assert list(detect_outliers(dirty).columns) == ["kolom", "jumlah_outlier", "persen"]


def test_build_eda_summary_is_json_serializable() -> None:
    summary = build_eda_summary(_df())
    text = json.dumps(summary, allow_nan=False)  # NaN/inf akan memicu error
    assert len(text) > 1000
    for key in ("deskriptif", "korelasi", "benford_per_kelompok", "distribusi_kelas", "perbandingan_kelompok"):
        assert key in summary
    n = len(summary["korelasi"]["kolom"])
    assert len(summary["korelasi"]["matriks"]) == n


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"OK  {t.__name__}")
    print(f"\n{len(tests)} uji lulus.")
