"""Uji resampling dan ResamplingClassifier. Jalankan: python -m tests.test_imbalance"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from src.data.split import group_stratified_cv
from src.models.imbalance import (
    ResamplingClassifier, class_counts, random_oversample, random_undersample, resample, smote,
)
from src.models.training import class_distribution_after
from src.models.zoo import IsolationForestScorer


def _toy(n_min: int = 20, n_maj: int = 180, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_min + n_maj, 3))
    y = np.r_[np.ones(n_min), np.zeros(n_maj)].astype(int)
    return X, y


def test_random_oversample() -> None:
    X, y = _toy()
    Xr, yr = random_oversample(X, y, 1.0, seed=1)
    assert class_counts(yr) == {"non_fraud": 180, "fraud": 180}
    assert np.array_equal(Xr[:len(X)], X)                                   # data asli utuh di depan
    minority = {tuple(r) for r in X[y == 1]}
    assert all(tuple(r) in minority for r in Xr[len(X):])                  # hanya salinan minoritas


def test_random_undersample() -> None:
    X, y = _toy()
    Xr, yr = random_undersample(X, y, 1.0, seed=1)
    assert class_counts(yr) == {"non_fraud": 20, "fraud": 20}
    assert {tuple(r) for r in Xr} <= {tuple(r) for r in X}                 # subset data asli
    assert (yr == 1).sum() == (y == 1).sum()                               # semua minoritas dipertahankan


def test_smote_synthetic_points_lie_between_minority_neighbors() -> None:
    rng = np.random.default_rng(0)
    xm = rng.uniform(0, 1, 15)
    Xmin = np.column_stack([xm, np.zeros(15)])                              # minoritas pada garis y = 0
    Xmaj = rng.normal(5, 1, size=(100, 2))
    X = np.vstack([Xmin, Xmaj]); y = np.r_[np.ones(15), np.zeros(100)].astype(int)
    Xr, yr = smote(X, y, 1.0, k_neighbors=5, seed=3)
    assert class_counts(yr) == {"non_fraud": 100, "fraud": 100}
    assert np.array_equal(Xr[:115], X)
    syn = Xr[115:]
    assert np.allclose(syn[:, 1], 0.0)                                      # tetap di garis minoritas
    assert syn[:, 0].min() >= xm.min() - 1e-12 and syn[:, 0].max() <= xm.max() + 1e-12   # di dalam hull
    assert len({tuple(r) for r in syn}) > 50                                # bukan sekadar duplikat


def test_ratio_and_determinism_and_edge_cases() -> None:
    X, y = _toy()
    _, yr = resample(X, y, "smote", ratio=0.5, seed=2)
    assert class_counts(yr)["fraud"] == 90
    a = resample(X, y, "smote", seed=7); b = resample(X, y, "smote", seed=7); c = resample(X, y, "smote", seed=8)
    assert np.array_equal(a[0], b[0]) and not np.array_equal(a[0], c[0])
    X1 = np.vstack([X[y == 0], X[y == 1][:1]]); y1 = np.r_[np.zeros(180), 1].astype(int)
    assert class_counts(smote(X1, y1, 1.0, seed=0)[1])["fraud"] == 180      # minoritas 1 -> fallback
    assert resample(X, y, "none")[0] is X and resample(X, y, "class_weight")[0] is X
    for bad in (lambda: resample(X, y, "xyz"), lambda: resample(X, y, "smote", ratio=0.0),
                lambda: resample(X, np.zeros(len(y), dtype=int), "smote"), lambda: resample(X, y + 1, "smote")):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("harus ValueError")


class _Spy(ClassifierMixin, BaseEstimator):
    """Merekam data yang dilihat saat fit."""
    def __init__(self, class_weight=None): self.class_weight = class_weight
    def fit(self, X, y):
        self.seen_ = np.asarray(X).copy(); self.classes_ = np.array([0, 1]); return self
    def predict_proba(self, X): return np.full((len(X), 2), 0.5)


def test_resampling_never_touches_validation_rows() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 4)); y = (rng.random(600) < 0.1).astype(int); g = np.repeat(np.arange(200), 3)
    for strat in ("random_over", "smote", "random_under"):
        for tr, te in group_stratified_cv(y, g, 4, seed=0):
            rc = ResamplingClassifier(_Spy(), strat, random_state=0).fit(X[tr], y[tr])
            seen = {tuple(r) for r in rc.estimator_.seen_}
            assert not any(tuple(r) in seen for r in X[te]), strat
            assert rc.n_train_original_["fraud"] == int(y[tr].sum())


def test_class_weight_strategy_and_errors() -> None:
    X, y = _toy()
    rf = ResamplingClassifier(RandomForestClassifier(n_estimators=10, random_state=0), "class_weight").fit(X, y)
    assert rf.estimator_.class_weight == "balanced" and rf.n_train_resampled_ == rf.n_train_original_
    for est, strat in ((IsolationForestScorer(), "class_weight"), (None, "none"), (_Spy(), "bogus")):
        try:
            ResamplingClassifier(est, strat).fit(X, y)
        except ValueError:
            continue
        raise AssertionError("harus ValueError")
    try:
        ResamplingClassifier(_Spy()).predict_proba(X)
    except RuntimeError:
        pass
    else:
        raise AssertionError("harus RuntimeError sebelum fit")
    c = clone(ResamplingClassifier(_Spy(), "smote", 0.5, 3, 9))
    assert (c.strategy, c.sampling_ratio, c.k_neighbors, c.random_state) == ("smote", 0.5, 3, 9)


def test_label_permutation_gives_chance_level_for_every_strategy() -> None:
    """Pada data TANPA sinyal, AUC validasi harus ~0.5. Jika resampling bocor, AUC akan melambung."""
    from src.evaluation.metrics import ranking_metrics
    from src.models.zoo import make_model
    rng = np.random.default_rng(0)
    n = 1500
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=list("abcde"))
    y = (rng.random(n) < 0.08).astype(int); g = np.repeat(np.arange(n // 3), 3)
    for model, strat in (("logreg", "smote"), ("logreg", "random_over"), ("random_forest", "random_over"),
                         ("random_forest", "smote"), ("logreg", "class_weight")):
        aucs = []
        for tr, te in group_stratified_cv(y, g, 5, seed=0):
            p = make_model(model, None, strat, 0).fit(X.iloc[tr], y[tr])
            aucs.append(ranking_metrics(y[te], p.predict_proba(X.iloc[te])[:, 1])["roc_auc"])
        assert 0.40 < np.mean(aucs) < 0.60, (model, strat, np.mean(aucs))


def test_resampling_before_split_leaks_and_the_detector_sees_it() -> None:
    """Kebalikan dari tes di atas: meresampling SEBELUM split menggelembungkan AUC pada data noise."""
    rng = np.random.default_rng(1)
    n = 900
    X = rng.normal(size=(n, 5)); y = (rng.random(n) < 0.08).astype(int)
    Xo, yo = random_oversample(X, y, 1.0, seed=0)                          # SALAH: oversample dulu
    idx = rng.permutation(len(yo)); cut = int(0.7 * len(yo))
    rf = RandomForestClassifier(100, random_state=0, n_jobs=-1).fit(Xo[idx[:cut]], yo[idx[:cut]])
    auc = roc_auc_score(yo[idx[cut:]], rf.predict_proba(Xo[idx[cut:]])[:, 1])
    assert auc > 0.8, auc                                                    # kebocoran terdeteksi (padahal tanpa sinyal)


def test_class_distribution_table() -> None:
    _, y = _toy()
    t = class_distribution_after(y).set_index("strategi")
    assert t.loc["none", "fraud_sesudah"] == 20 and t.loc["smote", "fraud_sesudah"] == 180
    assert t.loc["random_under", "non_fraud_sesudah"] == 20 and "balanced" in t.loc["class_weight", "catatan"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"OK  {t.__name__}")
    print(f"\n{len(tests)} uji lulus.")
