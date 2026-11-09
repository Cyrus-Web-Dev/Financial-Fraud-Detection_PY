"""Uji diagnostik regresi logistik. Jalankan: python -m tests.test_diagnostics"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from src.evaluation.diagnostics import box_tidwell, hosmer_lemeshow, influence_diagnostics, influence_summary
from src.models.logit import LogitInference


def _sim(n: int = 4000, seed: int = 0, quad: float = 0.0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    logit = -2.2 + 1.0 * X.a - 0.6 * X.b + quad * X.a ** 2
    return X, (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)


def test_hosmer_lemeshow_hand_computed() -> None:
    # 40 observasi, 4 kelompok @10 (p seragam dalam kelompok); statistik dihitung manual per kelompok
    pg1, pg2 = np.repeat([.1, .2], 10), np.repeat([.7, .9], 10)
    p = np.r_[pg1, pg2]; y = np.r_[np.r_[np.ones(3), np.zeros(17)], np.r_[np.ones(15), np.zeros(5)]].astype(int)
    r = hosmer_lemeshow(y, p, groups=4)
    t = r["tabel"]
    assert t["n"].tolist() == [10, 10, 10, 10] and r["dof"] == 2
    exp = t["fraud_diharapkan"].to_numpy(); obs = t["fraud_teramati"].to_numpy()
    manual = sum((o - e) ** 2 / (e * (1 - e / 10)) for o, e in zip(obs, exp))
    assert math.isclose(r["statistik"], manual) and math.isclose(r["p_value"], stats.chi2.sf(manual, 2))


def test_hosmer_lemeshow_calibrated_vs_misspecified() -> None:
    X, y = _sim(n=6000, seed=1)
    good = LogitInference().fit(X, y)
    ok = hosmer_lemeshow(y, good.predict_proba(X)[:, 1])
    assert ok["p_value"] > 0.05 and ok["dof"] == 8
    # model SALAH: efek kuadratik kuat pada a diabaikan
    Xq, yq = _sim(n=6000, seed=1, quad=0.9)
    bad = LogitInference().fit(Xq, yq)
    assert hosmer_lemeshow(yq, bad.predict_proba(Xq)[:, 1])["p_value"] < 0.01
    # tingkat penolakan di bawah H0 mendekati alpha
    rej = []
    for seed in range(150):
        Xs, ys = _sim(n=1500, seed=100 + seed)
        m = LogitInference().fit(Xs, ys)
        rej.append(hosmer_lemeshow(ys, m.predict_proba(Xs)[:, 1])["p_value"] < 0.05)
    assert 0.01 < np.mean(rej) < 0.12, np.mean(rej)
    for bad_call in (lambda: hosmer_lemeshow(y[:30], good.predict_proba(X)[:30, 1]), lambda: hosmer_lemeshow(y, y[:-1], 10),
                     lambda: hosmer_lemeshow(y, y, 2)):
        try:
            bad_call()
        except ValueError:
            continue
        raise AssertionError("harus ValueError")


def test_box_tidwell_flags_nonlinearity_only() -> None:
    rng = np.random.default_rng(0)
    n = 6000
    X = pd.DataFrame({"lin": rng.lognormal(0, 0.4, n), "logx": rng.lognormal(0, 0.8, n), "noise": rng.normal(size=n)})
    logit = -1.5 + 1.2 * X["lin"] + 1.5 * np.log(X["logx"])          # logit linear pada lin, LINEAR DALAM LOG pada logx
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    r = box_tidwell(X, y).set_index("fitur")
    assert r.loc["logx", "linear"] is np.False_ or not r.loc["logx", "linear"]
    assert r.loc["logx", "p_value"] < 1e-4
    assert r.loc["lin", "p_value"] > 0.001 and r.loc["noise", "p_value"] > 0.001
    assert bool(r.loc["noise", "digeser"]) and not bool(r.loc["lin", "digeser"])      # noise punya nilai <= 0 -> digeser
    assert np.allclose(r["ambang_bonferroni"], 0.05 / 3)
    assert list(box_tidwell(X, y, features=["lin"])["fitur"]) == ["lin"]


def test_influence_identities() -> None:
    X, y = _sim(n=2500, seed=2)
    m = LogitInference().fit(X, y)
    d = influence_diagnostics(m, X, y)
    k = m.n_params_
    assert math.isclose(d["leverage"].sum(), k, rel_tol=1e-9)                          # trace matriks topi = k
    assert math.isclose((d["resid_deviance"] ** 2).sum(), -2 * m.llf_, rel_tol=1e-9)   # jumlah kuadrat deviance = -2 lnL
    p = d["prob"]
    assert math.isclose((d["resid_pearson"] ** 2).sum(), (((y - p) ** 2) / (p * (1 - p))).sum(), rel_tol=1e-9)
    assert (d["leverage"] > 0).all() and (d["leverage"] < 1).all() and (d["cooks_d"] >= 0).all()
    assert np.allclose(d["cooks_d"], d["resid_pearson"] ** 2 * d["leverage"] / (k * (1 - d["leverage"]) ** 2))
    assert np.allclose(np.sign(d["resid_deviance"]), np.sign(y - p))
    s = influence_summary(d, k)
    assert s["n"] == 2500 and s["cooks_d_maks"] == d["cooks_d"].max() and 0 < s["persen_resid_deviance_gt_2"] < 15


def test_cooks_distance_matches_leave_one_out_refit_ranking() -> None:
    """Aproksimasi satu langkah harus sepakat dengan fit ulang tanpa observasi i (peringkat)."""
    X, y = _sim(n=600, seed=3)
    full = LogitInference().fit(X, y)
    d = influence_diagnostics(full, X, y)
    A = full._design(X)
    info = A.T @ (A * (d["prob"] * (1 - d["prob"])).to_numpy()[:, None])
    top = d["cooks_d"].sort_values(ascending=False).index[:25].tolist() + list(range(0, 600, 40))
    exact = []
    for i in top:
        keep = np.arange(600) != i
        b_i = LogitInference().fit(X[keep], y[keep]).params_.to_numpy()
        delta = full.params_.to_numpy() - b_i
        exact.append(float(delta @ info @ delta / full.n_params_))
    rho = stats.spearmanr(d.loc[top, "cooks_d"], exact).statistic
    assert rho > 0.95, rho
    assert np.allclose(np.array(exact), d.loc[top, "cooks_d"].to_numpy(), rtol=0.35, atol=1e-4)


def test_outlier_has_highest_influence() -> None:
    X, y = _sim(n=800, seed=4)
    X = X.copy(); X.loc[0, "a"] = 12.0; y = y.copy(); y[0] = 0                          # leverage tinggi dan label berlawanan
    d = influence_diagnostics(LogitInference().fit(X, y), X, y)
    assert d["cooks_d"].idxmax() == 0                                  # paling berpengaruh menurut Cook's D
    assert d.loc[0, "resid_deviance"] < -3                             # dan tampak sebagai residual deviance ekstrem
    # Catatan: leverage pada regresi logistik TIDAK naik monoton terhadap nilai ekstrem karena bobot
    # w = p(1-p) mengecil saat p -> 1; titik ini justru ber-leverage kecil. Cook's D menangkap gabungan keduanya.
    assert d.loc[0, "leverage"] < d["leverage"].median() * 2


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"OK  {t.__name__}")
    print(f"\n{len(tests)} uji lulus.")
