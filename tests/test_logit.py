"""Uji LogitInference (IRLS/Firth). Jalankan: python -m tests.test_logit"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

from src.models.logit import LogitInference, add_unit_odds_ratios, likelihood_ratio_test


def _data(n: int = 3000, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])          # c = noise murni
    p = 1 / (1 + np.exp(-(-2.5 + 1.0 * X.a - 0.7 * X.b)))
    return X, (rng.random(n) < p).astype(int)


def test_matches_sklearn_unpenalized() -> None:
    X, y = _data()
    m = LogitInference().fit(X, y)
    sk = LogisticRegression(C=1e12, solver="newton-cg", tol=1e-12, max_iter=1000).fit(X, y)
    ref = np.r_[sk.intercept_, sk.coef_.ravel()]
    assert np.abs(m.params_.to_numpy() - ref).max() < 1e-6
    assert m.converged_ and not m.separation_suspected_ and m.n_iter_ < 15
    assert np.allclose(m.predict_proba(X)[:, 1], sk.predict_proba(X)[:, 1], atol=1e-6)


def test_standard_errors_match_numerical_hessian() -> None:
    X, y = _data(n=1500, seed=1)
    m = LogitInference().fit(X, y)
    A = np.column_stack([np.ones(len(X)), X.to_numpy()])

    def ll(b):
        eta = A @ b
        return np.sum(y * -np.logaddexp(0, -eta) + (1 - y) * -np.logaddexp(0, eta))

    b0, h, k = m.params_.to_numpy(), 1e-4, 4
    H = np.zeros((k, k)); I = np.eye(k) * h
    for i in range(k):
        for j in range(k):
            H[i, j] = (ll(b0 + I[i] + I[j]) - ll(b0 + I[i] - I[j]) - ll(b0 - I[i] + I[j]) + ll(b0 - I[i] - I[j])) / (4 * h * h)
    assert np.abs(np.sqrt(np.diag(np.linalg.inv(-H))) - m.bse_.to_numpy()).max() < 1e-6


def test_summary_and_fit_statistics_are_internally_consistent() -> None:
    X, y = _data()
    m = LogitInference().fit(X, y)
    s, f = m.summary(), m.model_fit()
    assert np.allclose(s["odds_ratio"], np.exp(s["koefisien"]))
    assert np.allclose(s["z"], s["koefisien"] / s["std_error"])
    assert np.allclose(s["p_value"], 2 * stats.norm.sf(np.abs(s["z"])))
    assert np.allclose(np.log(s["or_ci_atas"]) - s["koefisien"], s["koefisien"] - np.log(s["or_ci_bawah"]))
    assert math.isclose(s.loc["a", "ci_atas"] - s.loc["a", "koefisien"], 1.959964 * s.loc["a", "std_error"], rel_tol=1e-5)
    # efek sejati signifikan, noise tidak; arah benar
    assert s.loc["a", "p_value"] < 1e-10 and s.loc["a", "odds_ratio"] > 1
    assert s.loc["b", "p_value"] < 1e-10 and s.loc["b", "odds_ratio"] < 1 and s.loc["c", "p_value"] > 0.01
    k, n = f["n_parameter"], f["n_obs"]
    assert math.isclose(f["aic"], -2 * f["log_likelihood"] + 2 * k) and math.isclose(f["bic"], -2 * f["log_likelihood"] + k * math.log(n))
    assert math.isclose(f["pseudo_r2_mcfadden"], 1 - f["log_likelihood"] / f["log_likelihood_null"])
    assert math.isclose(f["lr_chi2"], 2 * (f["log_likelihood"] - f["log_likelihood_null"]))
    assert f["lr_df"] == 3 and f["lr_p_value"] < 1e-10
    # sifat MLE dengan intersep: rata-rata probabilitas fit = proporsi fraud observasi
    assert math.isclose(m.predict_proba(X)[:, 1].mean(), y.mean(), abs_tol=1e-8)


def test_firth_intercept_only_matches_closed_form() -> None:
    for k, n in [(7, 100), (1, 40), (20, 60)]:
        y = np.r_[np.ones(k), np.zeros(n - k)]
        firth = LogitInference(firth=True).fit(np.empty((n, 0)), y)
        mle = LogitInference().fit(np.empty((n, 0)), y)
        assert math.isclose(firth.params_["const"], math.log((k + 0.5) / (n - k + 0.5)), abs_tol=1e-6)
        assert math.isclose(mle.params_["const"], math.log(k / (n - k)), abs_tol=1e-6)


def test_separation_detected_and_firth_gives_finite_estimates() -> None:
    xs = np.r_[np.linspace(-3, -0.1, 40), np.linspace(0.1, 3, 40)]
    X, y = pd.DataFrame({"x": xs}), (xs > 0).astype(int)
    plain, firth = LogitInference().fit(X, y), LogitInference(firth=True).fit(X, y)
    assert plain.separation_suspected_ and abs(plain.params_["x"]) > 15
    assert firth.converged_ and not firth.separation_suspected_ and 0 < firth.params_["x"] < 15
    assert np.isfinite(firth.bse_).all()


def test_firth_reduces_small_sample_bias() -> None:
    """Kejadian langka + sampel kecil: MLE melebih-lebihkan |slope|; Firth mengurangi bias dan RMSE."""
    rng = np.random.default_rng(0)
    mle_b, firth_b = [], []
    for _ in range(400):
        x = rng.normal(size=80)
        y = (rng.random(80) < 1 / (1 + np.exp(-(-3.0 + 1.0 * x)))).astype(int)
        if y.sum() < 3:
            continue
        X = pd.DataFrame({"x": x})
        mle_b.append(LogitInference(max_iter=200).fit(X, y).params_["x"])
        firth_b.append(LogitInference(firth=True).fit(X, y).params_["x"])
    mle_b, firth_b = np.array(mle_b), np.array(firth_b)
    assert abs(firth_b.mean() - 1.0) < abs(mle_b.mean() - 1.0)                        # bias lebih kecil
    assert mle_b.mean() > 1.05                                                         # MLE memang bias ke atas
    assert np.sqrt(np.mean((firth_b - 1) ** 2)) < np.sqrt(np.mean((mle_b - 1) ** 2))   # RMSE lebih kecil


def test_likelihood_ratio_test_nested_models() -> None:
    X, y = _data()
    full = LogitInference().fit(X, y)
    no_a = LogitInference().fit(X[["b", "c"]], y)
    no_c = LogitInference().fit(X[["a", "b"]], y)
    assert likelihood_ratio_test(full, no_a)["p_value"] < 1e-10                  # a berpengaruh
    assert likelihood_ratio_test(full, no_c)["p_value"] > 0.01                   # c noise
    assert likelihood_ratio_test(full, no_a)["df"] == 1
    for args in ((no_a, full), (full, LogitInference().fit(X.iloc[:-5], y[:-5]))):
        try:
            likelihood_ratio_test(*args)
        except ValueError:
            continue
        raise AssertionError("harus ValueError")


def test_unit_odds_ratio_equals_model_fit_on_raw_units() -> None:
    """MLE ekuivarian terhadap skala linear: exp(b_z / sd) harus sama dengan OR model pada satuan asli."""
    rng = np.random.default_rng(2)
    raw = pd.DataFrame({"u": rng.normal(1.0, 0.2, 2500), "v": rng.normal(50, 15, 2500)})
    y = (rng.random(2500) < 1 / (1 + np.exp(-(-6 + 2.0 * raw.u + 0.05 * raw.v)))).astype(int)
    sd = raw.std(ddof=0)
    z = (raw - raw.mean()) / sd
    on_z = add_unit_odds_ratios(LogitInference().fit(z, y).summary(), sd)
    on_raw = LogitInference().fit(raw, y).summary()
    for v in ("u", "v"):
        assert math.isclose(on_z.loc[v, "or_per_satuan"], on_raw.loc[v, "odds_ratio"], rel_tol=1e-5)
        assert math.isclose(on_z.loc[v, "or_satuan_ci_bawah"], on_raw.loc[v, "or_ci_bawah"], rel_tol=1e-4)
    assert np.isnan(on_z.loc["const", "or_per_satuan"])


def test_input_validation_and_errors() -> None:
    X, y = _data(n=100)
    bad_calls = [
        lambda: LogitInference().fit(X.iloc[:-1], y),
        lambda: LogitInference().fit(X.assign(a=np.nan), y),
        lambda: LogitInference().fit(X, np.zeros(100)),
        lambda: LogitInference().fit(X, y + 1),
        lambda: LogitInference().fit(X.rename(columns={"a": "const"}), y),
        lambda: LogitInference().fit(X.iloc[:3], y[:3]),
    ]
    for call in bad_calls:
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("harus ValueError")
    try:
        LogitInference().predict_proba(X)
    except RuntimeError:
        pass
    else:
        raise AssertionError("harus RuntimeError")
    m = LogitInference().fit(X, y)
    assert np.allclose(m.predict_proba(X[["c", "b", "a"]]), m.predict_proba(X))   # urutan kolom tidak berpengaruh
    try:
        m.predict_proba(X[["a", "b"]])
    except ValueError:
        pass
    else:
        raise AssertionError("kolom hilang harus ditolak")


def test_matches_statsmodels_when_available() -> None:
    try:
        import statsmodels.api as sm
    except ImportError:
        print("      (dilewati: statsmodels tidak terpasang)")
        return
    X, y = _data()
    m = LogitInference().fit(X, y)
    ref = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
    assert np.allclose(m.params_.to_numpy(), ref.params.to_numpy(), atol=1e-6)
    assert np.allclose(m.bse_.to_numpy(), ref.bse.to_numpy(), atol=1e-6)
    assert math.isclose(m.llf_, ref.llf, rel_tol=1e-9) and math.isclose(m.aic_, ref.aic, rel_tol=1e-9)
    assert math.isclose(m.bic_, ref.bic, rel_tol=1e-9) and math.isclose(m.pseudo_r2_, ref.prsquared, rel_tol=1e-9)
    assert math.isclose(m.lr_stat_, ref.llr, rel_tol=1e-9)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"OK  {t.__name__}")
    print(f"\n{len(tests)} uji lulus.")
