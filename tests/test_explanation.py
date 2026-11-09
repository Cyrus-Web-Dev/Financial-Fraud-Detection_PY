"""Uji penjelasan model (Shapley, importance, odds). Jalankan: python -m tests.test_explanation"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.evaluation.metrics import ranking_metrics
from src.explanation.importance import (
    beeswarm_data, dependence_data, mean_abs_shap, permutation_importance, rank_agreement,
)
from src.explanation.local import explain_instance
from src.explanation.odds import interpret_odds_ratios, linear_contributions
from src.explanation.shapley import ExactShapleyExplainer, linear_shap, model_scorer
from src.models.logit import LogitInference, add_unit_odds_ratios


def _brute_force_shapley(f, x, B):
    """Rata-rata kontribusi marjinal atas SEMUA d! permutasi (definisi asli Shapley)."""
    d = len(x)
    value = lambda S: float(np.mean([f(np.where(np.isin(np.arange(d), list(S)), x, b)[None, :])[0] for b in B]))
    phi = np.zeros(d)
    perms = list(itertools.permutations(range(d)))
    for perm in perms:
        S = []
        for j in perm:
            before = value(S); S.append(j); phi[j] += value(S) - before
    return phi / len(perms)


def _nonlinear(A):          # fungsi nonlinear dengan interaksi; fitur 3 tidak dipakai (dummy)
    return np.sin(A[:, 0]) + A[:, 1] * A[:, 2] + 0.5 * (A[:, 0] > 0.2) * A[:, 1]


# -------------------------------- Shapley ---------------------------------
def test_matches_brute_force_over_all_permutations() -> None:
    rng = np.random.default_rng(0)
    B, X = rng.normal(size=(6, 4)), rng.normal(size=(3, 4))
    ex = ExactShapleyExplainer(_nonlinear, B, list("abcd"))
    for x in X:
        assert np.allclose(ex.shap_values_one(x), _brute_force_shapley(_nonlinear, x, B), atol=1e-10)


def test_efficiency_dummy_and_hand_computed_interaction() -> None:
    rng = np.random.default_rng(1)
    B, X = rng.normal(size=(20, 4)), rng.normal(size=(5, 4))
    ex = ExactShapleyExplainer(_nonlinear, B, list("abcd"))
    phi = ex.shap_values(X)
    assert np.allclose(phi.sum(axis=1) + ex.expected_value, _nonlinear(X), atol=1e-10)      # efisiensi
    assert np.allclose(phi[:, 3], 0.0, atol=1e-12)                                          # dummy
    # f = x0 * x1; latar: (1, 2), (3, 4); x = (5, 7)
    f = lambda A: A[:, 0] * A[:, 1]
    ex2 = ExactShapleyExplainer(f, np.array([[1.0, 2.0], [3.0, 4.0]]), ["u", "w"])
    v0, v_u, v_w, v_uw = (1 * 2 + 3 * 4) / 2, 5 * (2 + 4) / 2, 7 * (1 + 3) / 2, 35.0
    expect = [0.5 * ((v_u - v0) + (v_uw - v_w)), 0.5 * ((v_w - v0) + (v_uw - v_u))]
    assert np.allclose(ex2.shap_values_one(np.array([5.0, 7.0])), expect) and math.isclose(ex2.expected_value, v0)


def test_symmetry_and_additivity() -> None:
    rng = np.random.default_rng(2)
    col = rng.normal(size=15)
    B = np.column_stack([col, col, rng.normal(size=15)])                                    # fitur 0 dan 1 identik di latar
    sym = lambda A: A[:, 0] * A[:, 1] + A[:, 2]
    phi = ExactShapleyExplainer(sym, B, list("abc")).shap_values_one(np.array([0.7, 0.7, -0.3]))
    assert math.isclose(phi[0], phi[1], abs_tol=1e-12)
    f, g = _nonlinear, lambda A: np.cos(A[:, 2]) - A[:, 3] ** 2
    B4, x = rng.normal(size=(10, 4)), rng.normal(size=4)
    pf = ExactShapleyExplainer(f, B4, list("abcd")).shap_values_one(x)
    pg = ExactShapleyExplainer(g, B4, list("abcd")).shap_values_one(x)
    pfg = ExactShapleyExplainer(lambda A: f(A) + g(A), B4, list("abcd")).shap_values_one(x)
    assert np.allclose(pfg, pf + pg, atol=1e-10)


def test_linear_models_closed_form_and_logit_agreement() -> None:
    rng = np.random.default_rng(3)
    n = 1500
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    y = (rng.random(n) < 1 / (1 + np.exp(-(-2 + 1.0 * X.a - 0.7 * X.b + 0.4 * X.c)))).astype(int)
    m = LogitInference().fit(X, y)
    B, T = X.iloc[:30].to_numpy(), X.iloc[100:106].to_numpy()
    coef = m.params_.drop("const").to_numpy()
    f = lambda A: A @ coef + m.params_["const"]                                             # skala log-odds
    ex = ExactShapleyExplainer(f, B, list("abcd"))
    phi_exact = ex.shap_values(T)
    phi_lin, offset = linear_shap(coef, T, B)
    assert np.allclose(phi_exact, phi_lin, atol=1e-10)                                      # dua implementasi sepakat
    assert math.isclose(ex.expected_value, m.params_["const"] + offset, abs_tol=1e-10)
    via_model = ExactShapleyExplainer(lambda A: m.decision_function(pd.DataFrame(A, columns=list("abcd"))), B, list("abcd"))
    assert np.allclose(via_model.shap_values(T), phi_lin, atol=1e-10)
    for bad in (lambda: linear_shap(coef[:2], T, B),):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError


def test_random_forest_efficiency_and_scorer_output() -> None:
    rng = np.random.default_rng(4)
    X = rng.normal(size=(500, 5)); y = (X[:, 0] + 0.5 * X[:, 1] * X[:, 2] + rng.normal(scale=0.7, size=500) > 1).astype(int)
    rf = RandomForestClassifier(60, min_samples_leaf=5, random_state=0).fit(X, y)
    for output in ("proba", "logit"):
        fn = model_scorer(rf, output)
        ex = ExactShapleyExplainer(fn, X[:25], list("abcde"))
        T = X[300:304]
        phi = ex.shap_values(T)
        assert np.allclose(phi.sum(axis=1) + ex.expected_value, fn(T), atol=1e-9), output
    imp = mean_abs_shap(ExactShapleyExplainer(model_scorer(rf), X[:25], list("abcde")).shap_values(X[300:330]), list("abcde"))
    assert imp.loc[0, "fitur"] == "a" and math.isclose(imp["porsi"].sum(), 1.0)
    assert imp["peringkat"].tolist() == [1, 2, 3, 4, 5] and (imp["mean_abs_shap"].diff().dropna() <= 1e-15).all()
    try:
        model_scorer(rf, "xyz")
    except ValueError:
        pass
    else:
        raise AssertionError


def test_explainer_validation() -> None:
    B = np.zeros((5, 3))
    f = lambda A: A.sum(axis=1)
    bad_calls = [lambda: ExactShapleyExplainer(f, B, ["a", "b"]), lambda: ExactShapleyExplainer(f, np.zeros((5, 15)), list(range(15))),
                 lambda: ExactShapleyExplainer(f, np.full((5, 3), np.nan), list("abc")), lambda: ExactShapleyExplainer(f, B[:0], list("abc")),
                 lambda: ExactShapleyExplainer(f, B, list("abc")).shap_values(np.zeros((2, 4))),
                 lambda: ExactShapleyExplainer(lambda A: A.sum(axis=1)[:-1], B, list("abc"))]
    for call in bad_calls:
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("harus ValueError")


def test_matches_shap_library_when_available() -> None:
    try:
        import shap
    except ImportError:
        print("      (dilewati: shap tidak terpasang)")
        return
    rng = np.random.default_rng(5)
    X = rng.normal(size=(400, 4)); y = (X[:, 0] + X[:, 1] * X[:, 2] + rng.normal(scale=0.5, size=400) > 0.5).astype(int)
    rf = RandomForestClassifier(30, min_samples_leaf=5, random_state=0).fit(X, y)
    B, T = X[:20], X[300:304]
    ours = ExactShapleyExplainer(model_scorer(rf), B, list("abcd")).shap_values(T)
    ref = shap.TreeExplainer(rf, data=B, feature_perturbation="interventional", model_output="probability").shap_values(T)
    ref = np.asarray(ref[1] if isinstance(ref, list) else (ref[..., 1] if np.ndim(ref) == 3 else ref))
    assert np.allclose(ours, ref, atol=1e-6)


# ------------------------------- importance --------------------------------
def test_permutation_importance() -> None:
    rng = np.random.default_rng(6)
    n = 1200
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["sinyal", "lemah", "noise"])
    y = (rng.random(n) < 1 / (1 + np.exp(-(-1.5 + 2.0 * X.sinyal + 0.5 * X.lemah)))).astype(int)
    m = LogitInference().fit(X, y)
    fn = lambda D: m.predict_proba(D)[:, 1]
    metric = lambda a, b: ranking_metrics(a, b)["roc_auc"]
    imp = permutation_importance(fn, X, y, metric, n_repeats=15, seed=0)
    s = imp.set_index("fitur")["penurunan_mean"]
    assert imp.loc[0, "fitur"] == "sinyal" and s["sinyal"] > s["lemah"] > s["noise"] - 1e-9 and abs(s["noise"]) < 0.01
    assert math.isclose(imp["metrik_dasar"].iloc[0], metric(y, fn(X)))
    assert permutation_importance(fn, X, y, metric, 5, 7).equals(permutation_importance(fn, X, y, metric, 5, 7))
    assert (fn(X) == m.predict_proba(X)[:, 1]).all()                                        # X asli tidak termutasi


def test_dependence_data_picks_interacting_color_feature() -> None:
    rng = np.random.default_rng(7)
    n = 800
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["x", "mod", "bising"])
    phi = np.column_stack([X["x"] * (1 + 2 * (X["mod"] > 0)), 0 * X["x"], 0 * X["x"]]) + rng.normal(scale=0.05, size=(n, 3))
    d = dependence_data(phi, X, "x")
    assert d["warna_fitur"] == "mod" and len(d["x"]) == len(d["shap"]) == len(d["nilai_warna"]) == n
    assert dependence_data(phi, X, "x", color="bising")["warna_fitur"] == "bising"
    b = beeswarm_data(phi, X, max_points_per_feature=100)
    assert len(b) == 300 and b["persentil_nilai"].between(0, 1).all() and set(b["fitur"]) == {"x", "mod", "bising"}


def test_rank_agreement() -> None:
    a = pd.Series({"f1": 5.0, "f2": 3.0, "f3": 1.0, "f4": 0.5})
    r = rank_agreement({"A": a, "B": a * 10, "C": a[::-1].reset_index(drop=True).set_axis(a.index)}, top_k=2)
    assert math.isclose(r["spearman"].loc["A", "B"], 1.0) and r["spearman"].loc["A", "C"] < 0
    assert r["irisan_top2"].loc["A", "B"] == 1.0 and r["irisan_top2"].loc["A", "C"] == 0.0 and r["top_fitur"]["A"] == ["f1", "f2"]


# ----------------------------- odds dan penjelasan --------------------------
def test_interpret_odds_ratios_theory_matching() -> None:
    rng = np.random.default_rng(8)
    n = 4000
    Z = pd.DataFrame(rng.normal(size=(n, 3)), columns=["dsri", "cash_ratio", "aqi"])
    # dsri: +, sesuai teori (+1) | cash_ratio: kenaikan menaikkan fraud, BERLAWANAN dengan teori (-1) | aqi: noise
    y = (rng.random(n) < 1 / (1 + np.exp(-(-2.5 + 0.8 * Z.dsri + 0.5 * Z.cash_ratio)))).astype(int)
    summ = add_unit_odds_ratios(LogitInference().fit(Z, y).summary(), Z.std(ddof=0))
    t = interpret_odds_ratios(summ).set_index("fitur")
    assert "const" not in t.index
    assert t.loc["dsri", "kesesuaian"] == "sesuai teori" and t.loc["dsri", "signifikan"] and "meningkatkan odds fraud" in t.loc["dsri", "interpretasi"]
    assert t.loc["cash_ratio", "kesesuaian"] == "BERLAWANAN dengan teori"
    assert not t.loc["aqi", "signifikan"] and "tidak berpengaruh signifikan" in t.loc["aqi", "interpretasi"]
    import re
    assert re.search(r"OR = \d+,\d{2}; CI 95% \d+,\d{2}-\d+,\d{2}; p", t.loc["dsri", "interpretasi"])     # koma desimal, bukan titik
    assert not re.search(r"\d\.\d", t.loc["dsri", "interpretasi"])
    protect = summ.copy(); protect.loc["dsri", ["odds_ratio", "or_ci_bawah", "or_ci_atas"]] = [0.5, 0.4, 0.6]
    low = interpret_odds_ratios(protect).set_index("fitur").loc["dsri"]
    assert "menurunkan odds fraud sebesar 50%" in low["interpretasi"] and low["kesesuaian"] == "BERLAWANAN dengan teori"


def test_marginal_result_wording_and_catalog_signs() -> None:
    from src.features.ratios import FEATURE_CATALOG
    cat = {f.name: f for f in FEATURE_CATALOG}
    assert cat["sgai"].expected_sign == 1 and cat["lvgi"].expected_sign == 1          # arah hipotesis, bukan koefisien estimasi
    summ = pd.DataFrame({"koefisien": [0.3], "std_error": [0.15], "z": [2.0], "p_value": [0.052], "ci_bawah": [0.0], "ci_atas": [0.6],
                         "odds_ratio": [1.35], "or_ci_bawah": [0.99], "or_ci_atas": [1.8]}, index=["tata"])
    t = interpret_odds_ratios(summ).iloc[0]
    assert "marginal" in t["interpretasi"] and "tidak signifikan" in t["interpretasi"] and not t["signifikan"]
    summ.loc["tata", "p_value"] = 0.4
    assert "tidak berpengaruh signifikan" in interpret_odds_ratios(summ).iloc[0]["interpretasi"]


def test_local_explanation() -> None:
    names = ["dsri", "tata", "sgi"]
    phi = np.array([0.10, -0.02, 0.03]); x = pd.Series({"dsri": 1.4, "tata": 0.01, "sgi": 1.1})
    ref = pd.DataFrame({"dsri": np.linspace(0.8, 1.3, 100), "tata": np.linspace(-0.1, 0.1, 100), "sgi": np.linspace(0.9, 1.3, 100)})
    e = explain_instance(phi, x, 0.06, names, ref, "probabilitas", top_n=2)
    assert math.isclose(e["prediksi"], 0.06 + phi.sum()) and e["kontribusi"].loc[0, "fitur"] == "dsri"
    assert math.isclose(e["kontribusi"]["kontribusi"].sum(), phi.sum()) and len(e["kalimat"]) == 2
    assert "menaikkan risiko" in e["kalimat"][0] and "bernilai 1,4" in e["kalimat"][0] and "persentil ke-100" in e["kalimat"][0] and "+10,0 poin persen" in e["kalimat"][0]
    assert "menurunkan risiko" in explain_instance(np.array([-0.5, 0.0, 0.0]), x, 0.0, names, None, "log-odds")["kalimat"][0]
    try:
        explain_instance(phi[:2], x, 0.0, names)
    except ValueError:
        pass
    else:
        raise AssertionError
    coef = pd.Series({"const": -2.0, "dsri": 0.7, "tata": 0.3})
    lc = linear_contributions(coef, pd.Series({"dsri": 2.0, "tata": -1.0}))
    assert lc.loc[0, "fitur"] == "dsri" and math.isclose(lc.loc[0, "kontribusi_logodds"], 1.4) and math.isclose(lc.loc[1, "kontribusi_logodds"], -0.3)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"OK  {t.__name__}")
    print(f"\n{len(tests)} uji lulus.")
