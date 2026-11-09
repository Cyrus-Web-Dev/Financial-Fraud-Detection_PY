"""Uji kalibrasi, inferensi (bootstrap/DeLong/McNemar), dan ambang. Jalankan: python -m tests.test_evaluation"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

from src.evaluation.calibration import (
    IsotonicCalibrator, PlattCalibrator, calibration_report, calibration_slope_intercept, calibration_table,
    expected_calibration_error,
)
from src.evaluation.inference import (
    cluster_bootstrap, delong_auc_ci, delong_components, delong_test, holm_correction, mcnemar_test,
    paired_cluster_bootstrap,
)
from src.evaluation.metrics import ranking_metrics, threshold_metrics
from src.evaluation.thresholds import (
    apply_threshold, bayes_threshold, cost_optimal_threshold, decision_curve, downsample_curve, max_f1_threshold,
    threshold_scan, youden_threshold,
)


def _calibrated(n: int = 20000, seed: int = 0):
    rng = np.random.default_rng(seed)
    p = np.clip(rng.beta(1, 12, n), 0.001, 0.95)
    return p, (rng.random(n) < p).astype(int)


# --------------------------------- kalibrasi --------------------------------
def test_calibration_table_and_ece_hand_computed() -> None:
    y = np.array([0, 0, 0, 1, 0, 1, 1, 1]); p = np.array([.1, .1, .2, .2, .7, .8, .8, .9])
    t = calibration_table(y, p, n_bins=2)
    assert t["n"].tolist() == [4, 4]
    assert np.allclose(t["rata_prediksi"], [0.15, 0.8]) and np.allclose(t["frekuensi_teramati"], [0.25, 0.75])
    assert math.isclose(expected_calibration_error(y, p, 2), (4 * 0.10 + 4 * 0.05) / 8)
    assert (t["ci95_bawah"] <= t["frekuensi_teramati"]).all() and (t["frekuensi_teramati"] <= t["ci95_atas"]).all()
    for bad in (lambda: calibration_table(y, p * 2), lambda: calibration_table(y, p[:-1]),
                lambda: calibration_table(y, np.r_[p[:-1], np.nan])):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("harus ValueError")


def test_calibration_slope_intercept_known_cases() -> None:
    p, y = _calibrated()
    ok = calibration_slope_intercept(y, p)
    assert abs(ok["slope"] - 1) < 0.1 and abs(ok["intersep"]) < 0.15 and abs(ok["citl"]) < 0.08
    assert ok["slope_ci_bawah"] < 1 < ok["slope_ci_atas"]
    shifted = calibration_slope_intercept(y, expit(logit(p) + 0.8))          # prediksi terlalu tinggi 0,8 logit
    assert math.isclose(shifted["citl"], -0.8, abs_tol=0.08)
    overconf = calibration_slope_intercept(y, expit(logit(p) * 1.8))         # terlalu ekstrem -> slope ~ 1/1.8
    assert math.isclose(overconf["slope"], 1 / 1.8, abs_tol=0.06)
    rep = calibration_report(y, p)
    assert 0 < rep["brier"] < 0.1 and rep["brier_skill_score"] > 0 and rep["ece"] < 0.01
    try:
        calibration_slope_intercept(np.zeros(10, dtype=int), np.full(10, 0.1))
    except ValueError:
        pass
    else:
        raise AssertionError("satu kelas harus ditolak")


def test_platt_and_isotonic_fix_miscalibration() -> None:
    p, y = _calibrated(40000, seed=1)
    bad = expit(logit(p) + 1.2)                                               # miskalibrasi sistematis
    train, test = slice(0, 20000), slice(20000, None)
    raw = calibration_report(y[test], bad[test])
    platt = PlattCalibrator().fit(bad[train], y[train]).predict(bad[test])
    iso = IsotonicCalibrator().fit(bad[train], y[train]).predict(bad[test])
    for fixed in (platt, iso):
        rep = calibration_report(y[test], fixed)
        assert rep["brier"] < raw["brier"] and rep["ece"] < raw["ece"] / 3 and abs(rep["citl"]) < 0.15
    assert abs(calibration_report(y[test], platt)["slope"] - 1) < 0.12
    s = np.sort(bad[test])
    assert np.all(np.diff(IsotonicCalibrator().fit(bad[train], y[train]).predict(s)) >= -1e-12)   # monoton
    assert ranking_metrics(y[test], platt)["roc_auc"] == ranking_metrics(y[test], bad[test])["roc_auc"]   # Platt monoton
    for cal in (PlattCalibrator(), IsotonicCalibrator()):
        try:
            cal.predict(np.array([0.5]))
        except RuntimeError:
            continue
        raise AssertionError("harus RuntimeError")
    try:
        PlattCalibrator("xyz")
    except ValueError:
        pass
    else:
        raise AssertionError


# ------------------------------------ DeLong --------------------------------
def _brute_delong(y, S):
    pos, neg = S[:, y == 1], S[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    k = S.shape[0]
    psi = lambda a, b: (a > b) + 0.5 * (a == b)
    v_pos = np.array([[np.mean(psi(pos[r, i], neg[r])) for i in range(m)] for r in range(k)])
    v_neg = np.array([[np.mean(psi(pos[r], neg[r, j])) for j in range(n)] for r in range(k)])
    return v_pos.mean(axis=1), np.atleast_2d(np.cov(v_pos)) / m + np.atleast_2d(np.cov(v_neg)) / n


def test_delong_matches_brute_force_including_ties() -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(70) < 0.3).astype(int)
    s1 = np.round(rng.normal(size=70) + 1.2 * y, 1)                          # pembulatan -> banyak seri
    s2 = np.round(rng.normal(size=70) + 0.6 * y + 0.5 * s1, 1)
    auc, cov = delong_components(y, np.vstack([s1, s2]))
    b_auc, b_cov = _brute_delong(y, np.vstack([s1, s2]))
    assert np.allclose(auc, b_auc, atol=1e-12) and np.allclose(cov, b_cov, atol=1e-12)
    assert math.isclose(auc[0], roc_auc_score(y, s1), abs_tol=1e-12)


def test_delong_test_behaviour() -> None:
    rng = np.random.default_rng(1)
    n = 3000
    y = (rng.random(n) < 0.15).astype(int)
    good = rng.normal(size=n) + 1.5 * y
    weak = rng.normal(size=n) + 0.3 * y
    same = delong_test(y, good, good)
    assert same["p_value"] == 1.0 and same["selisih"] == 0
    r = delong_test(y, good, weak)
    assert r["p_value"] < 1e-6 and r["selisih"] > 0.15 and r["ci_bawah"] > 0
    assert math.isclose(delong_test(y, weak, good)["selisih"], -r["selisih"]) and \
        math.isclose(delong_test(y, weak, good)["p_value"], r["p_value"], rel_tol=1e-9)
    # Galat tipe I: dua model berkekuatan SAMA dengan derau independen -> laju penolakan H0 ~ alpha
    rejects = []
    for seed in range(300):
        g = np.random.default_rng(seed); yy = (g.random(500) < 0.2).astype(int)
        a, b = g.normal(size=500) + 1.2 * yy, g.normal(size=500) + 1.2 * yy
        rejects.append(delong_test(yy, a, b)["p_value"] < 0.05)
    assert 0.02 < np.mean(rejects) < 0.09, np.mean(rejects)
    ci = delong_auc_ci(y, good)
    assert ci["ci_bawah"] < ci["auc"] < ci["ci_atas"]
    try:
        delong_components(np.array([0, 0, 0, 1]), np.arange(4.0))
    except ValueError:
        pass
    else:
        raise AssertionError("kelas dengan < 2 observasi harus ditolak")


# --------------------------------- bootstrap --------------------------------
def test_cluster_bootstrap_is_wider_than_row_bootstrap_for_clustered_data() -> None:
    rng = np.random.default_rng(0)
    n_c, size = 120, 5
    yc = (rng.random(n_c) < 0.3).astype(int); sc = rng.normal(size=n_c) + 1.0 * yc
    y, s = np.repeat(yc, size), np.repeat(sc, size) + rng.normal(scale=0.05, size=n_c * size)
    g = np.repeat(np.arange(n_c), size)
    auc = lambda a, b: roc_auc_score(a, b)
    cl = cluster_bootstrap(y, s, g, auc, n_boot=400, seed=1)
    row = cluster_bootstrap(y, s, np.arange(len(y)), auc, n_boot=400, seed=1)
    assert cl["ci_bawah"] < cl["estimasi"] < cl["ci_atas"]
    assert (cl["ci_atas"] - cl["ci_bawah"]) > 1.5 * (row["ci_atas"] - row["ci_bawah"])
    assert cluster_bootstrap(y, s, g, auc, n_boot=100, seed=3) == cluster_bootstrap(y, s, g, auc, n_boot=100, seed=3)


def test_bootstrap_ci_agrees_with_delong_on_independent_data() -> None:
    rng = np.random.default_rng(2)
    n = 1500
    y = (rng.random(n) < 0.2).astype(int); s = rng.normal(size=n) + 1.0 * y
    b = cluster_bootstrap(y, s, np.arange(n), lambda a, c: roc_auc_score(a, c), n_boot=500, seed=0)
    d = delong_auc_ci(y, s)
    assert abs((b["ci_atas"] - b["ci_bawah"]) - (d["ci_atas"] - d["ci_bawah"])) < 0.25 * (d["ci_atas"] - d["ci_bawah"])


def test_paired_bootstrap_detects_difference() -> None:
    rng = np.random.default_rng(3)
    n = 1200
    y = (rng.random(n) < 0.2).astype(int); g = np.arange(n)
    good, weak = rng.normal(size=n) + 1.6 * y, rng.normal(size=n) + 0.2 * y
    auc = lambda a, b: roc_auc_score(a, b)
    r = paired_cluster_bootstrap(y, good, weak, g, auc, n_boot=300, seed=0)
    assert r["ci_bawah"] > 0 and r["p_value"] < 0.05 and r["selisih"] > 0.2
    r0 = paired_cluster_bootstrap(y, good, good, g, auc, n_boot=100, seed=0)
    assert r0["selisih"] == 0 and r0["ci_bawah"] == 0 == r0["ci_atas"]


# ----------------------------- Holm dan McNemar -----------------------------
def test_holm_and_mcnemar() -> None:
    assert np.allclose(holm_correction([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    q = holm_correction([0.5, np.nan, 0.01]); assert np.isnan(q[1]) and math.isclose(q[2], 0.02) and math.isclose(q[0], 0.5)
    assert (holm_correction([0.001, 0.2]) >= np.array([0.001, 0.2])).all()
    y = np.ones(40, dtype=int)
    a = np.r_[np.ones(10), np.zeros(3), np.ones(27)].astype(int)            # b = 10 (A benar, B salah)
    b_ = np.r_[np.zeros(10), np.ones(3), np.ones(27)].astype(int)           # c = 3
    r = mcnemar_test(y, a, b_)
    assert (r["b"], r["c"]) == (10, 3) and r["metode"] == "eksak binomial"
    assert math.isclose(r["p_value"], stats.binomtest(3, 13, 0.5).pvalue)
    y2 = np.ones(100, dtype=int)
    a2 = np.r_[np.ones(30), np.zeros(10), np.ones(60)].astype(int); b2 = np.r_[np.zeros(30), np.ones(10), np.ones(60)].astype(int)
    r2 = mcnemar_test(y2, a2, b2)
    assert math.isclose(r2["statistik"], (20 - 1) ** 2 / 40) and math.isclose(r2["p_value"], stats.chi2.sf(361 / 40, 1))
    assert mcnemar_test(y, a, a)["p_value"] == 1.0


# ----------------------------------- ambang ---------------------------------
def test_threshold_scan_matches_threshold_metrics() -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.2).astype(int); s = np.round(rng.random(400) * 0.7 + 0.3 * y, 2)   # banyak seri
    t = threshold_scan(y, s, max_points=100000)
    for i in (0, 10, len(t) // 2, len(t) - 1):
        m = threshold_metrics(y, s, float(t.loc[i, "ambang"]))
        assert (m["tp"], m["fp"], m["tn"], m["fn"]) == tuple(int(t.loc[i, c]) for c in ("tp", "fp", "tn", "fn"))
    assert t["ambang"].is_monotonic_increasing
    assert t.iloc[0]["recall"] == 1.0 and t.iloc[0]["fp"] == (y == 0).sum()        # ambang terendah = tandai semua
    assert t.iloc[-1]["tp"] + t.iloc[-1]["fp"] == (s == s.max()).sum()             # ambang tertinggi = hanya skor maksimum


def test_optimal_thresholds_match_brute_force() -> None:
    rng = np.random.default_rng(1)
    y = (rng.random(500) < 0.15).astype(int); s = rng.normal(size=500) + 1.3 * y
    cands = np.unique(s)
    cost = lambda t: 10 * threshold_metrics(y, s, t)["fn"] + 1 * threshold_metrics(y, s, t)["fp"]
    best = min(cost(t) for t in cands)
    r = cost_optimal_threshold(y, s, 10, 1)
    assert math.isclose(r["biaya_total"], best) and math.isclose(cost(r["ambang"]), best)
    j = lambda t: (lambda m: m["recall"] + m["specificity"] - 1)(threshold_metrics(y, s, t))
    assert math.isclose(youden_threshold(y, s)["youden_j"], max(j(t) for t in cands))
    f1 = lambda t: (lambda m: 0 if np.isnan(m["f1"]) else m["f1"])(threshold_metrics(y, s, t))
    assert math.isclose(max_f1_threshold(y, s)["f1"], max(f1(t) for t in cands))
    assert r["biaya_total"] <= min(r["biaya_tanpa_model"], r["biaya_tandai_semua"])
    cheap_fn = cost_optimal_threshold(y, s, 1, 10)["ambang"]                      # FP mahal -> ambang lebih tinggi
    assert cheap_fn > r["ambang"]


def test_empirical_cost_threshold_converges_to_bayes_threshold_when_calibrated() -> None:
    p, y = _calibrated(300000, seed=2)
    for c_fn, c_fp in ((10, 1), (5, 1), (20, 1)):
        t_star = bayes_threshold(c_fn, c_fp)
        emp = cost_optimal_threshold(y, p, c_fn, c_fp)["ambang"]
        assert abs(emp - t_star) < 0.02, (c_fn, emp, t_star)
    assert math.isclose(bayes_threshold(10, 1), 1 / 11)
    for bad in (lambda: bayes_threshold(0, 1), lambda: cost_optimal_threshold(np.r_[1, 0], np.r_[.9, .1], -1, 1)):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError


def test_apply_threshold_and_decision_curve_hand_computed() -> None:
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0]); p = np.array([.9, .4, .6, .1, .1, .1, .1, .1, .1, .1])
    m = apply_threshold(y, p, 0.5, c_fn=10, c_fp=1)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1) and m["biaya_total"] == 11 and m["biaya_tanpa_model"] == 20
    assert math.isclose(m["persen_ditandai"], 0.2)
    assert math.isclose(m["biaya_tanpa_model_per_observasi"], 2.0) and math.isclose(m["penghematan_biaya_persen"], 45.0)
    d = decision_curve(y, p, np.array([0.5, 0.25]))
    # t=0.5: TP=1, FP=1 -> 1/10 - 1/10 * 1 = 0 ; tandai semua: 0.2 - 0.8 * 1 = -0.6
    assert math.isclose(d.loc[0, "net_benefit_model"], 0.0, abs_tol=1e-12) and math.isclose(d.loc[0, "net_benefit_tandai_semua"], -0.6)
    # t=0.25: p >= 0.25 menandai 0.9 (TP), 0.4 (TP), 0.6 (FP) -> TP=2, FP=1: 2/10 - 1/10 * (1/3); tandai semua: 0.2 - 0.8/3
    assert math.isclose(d.loc[1, "net_benefit_model"], 0.2 - 0.1 / 3) and math.isclose(d.loc[1, "net_benefit_tandai_semua"], 0.2 - 0.8 / 3)
    assert (d["net_benefit_tidak_ada"] == 0).all()
    try:
        decision_curve(y, p, np.array([0.0, 0.5]))
    except ValueError:
        pass
    else:
        raise AssertionError
    x, yy = downsample_curve(np.arange(1000.0), np.arange(1000.0) ** 2, 100)
    assert len(x) <= 100 and x[0] == 0 and x[-1] == 999 and len(x) == len(yy)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"OK  {t.__name__}")
    print(f"\n{len(tests)} uji lulus.")
