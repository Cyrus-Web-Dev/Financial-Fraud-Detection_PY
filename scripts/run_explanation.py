"""Tahap 6: explainability (odds ratio, Shapley/SHAP, permutation importance, penjelasan lokal).

    python -m scripts.run_explanation                   # 200 instance uji, 40 baris latar
    python -m scripts.run_explanation --n-explain 100 --n-background 25     # lebih cepat

Prasyarat: run_feature_engineering dan run_modeling sudah dijalankan.
Keluaran: data/processed/explanation_results.json
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, PROCESSED_DIR, RANDOM_SEED
from src.data.schema import TARGET_COLUMN
from src.evaluation.metrics import ranking_metrics
from src.explanation.importance import beeswarm_data, dependence_data, mean_abs_shap, permutation_importance, rank_agreement
from src.explanation.local import explain_instance
from src.explanation.odds import interpret_odds_ratios, linear_contributions
from src.explanation.shapley import ExactShapleyExplainer, linear_shap, model_scorer
from src.models.logit import add_unit_odds_ratios
from src.models.training import load_model_bundle
from src.utils import to_jsonable

TREE_MODELS = ("random_forest", "hist_gb")


def _log(msg: str) -> None:
    print(msg, flush=True)


def _read_json(name: str):
    with open(PROCESSED_DIR / name, encoding="utf-8") as fh:
        return json.load(fh)


def main(n_explain: int = 200, n_background: int = 40) -> None:
    t0 = time.time()
    for n in ("features.csv", "split.json", "feature_selection.json"):
        if not (PROCESSED_DIR / n).exists():
            raise SystemExit(f"{n} belum ada. Jalankan run_feature_engineering dan run_modeling dulu.")
    table = pd.read_csv(PROCESSED_DIR / "features.csv")
    split, selected = _read_json("split.json"), _read_json("feature_selection.json")["selected"]
    train = table.iloc[split["train_idx"]].reset_index(drop=True)
    test = table.iloc[split["test_idx"]].reset_index(drop=True)
    yte = test[TARGET_COLUMN].to_numpy(int)
    rng = np.random.default_rng(RANDOM_SEED)
    res: dict = {"seed": RANDOM_SEED, "fitur": selected, "n_dijelaskan": n_explain, "n_latar": n_background}

    # 1. Odds ratio + interpretasi teori (regresi logistik MLE dan Firth)
    bundle = joblib.load(MODELS_DIR / "logit_mle.joblib")
    logit, pre = bundle["model"], bundle["preprocessor"]
    firth = joblib.load(MODELS_DIR / "logit_firth.joblib")["model"]
    summ = add_unit_odds_ratios(logit.summary(), pre.std_)
    interp = interpret_odds_ratios(summ)
    cmp_firth = pd.DataFrame({"mle": logit.params_, "firth": firth.params_}).drop("const")
    cmp_firth["selisih"] = cmp_firth["firth"] - cmp_firth["mle"]
    res["odds_ratio"] = summ.reset_index()
    res["interpretasi_odds_ratio"] = interp
    res["mle_vs_firth"] = cmp_firth.reset_index(names="fitur")
    _log("[1] Interpretasi odds ratio (per 1 SD):")
    for _, r in interp.iterrows():
        _log(f"    - {r['interpretasi']}")
    _log(f"    Selisih koefisien MLE vs Firth (maks |beda|): {cmp_firth['selisih'].abs().max():.4f}")

    # 2. Sampel penjelasan dan latar (dari data latih) -- semua di ruang fitur terpreproses
    idx = np.sort(rng.choice(len(test), size=min(n_explain, len(test)), replace=False))
    X_orig = test[selected].iloc[idx].reset_index(drop=True)
    y_sample = yte[idx]
    bg_rows = train[selected].iloc[rng.choice(len(train), size=n_background, replace=False)]
    Z_sample, Z_bg = pre.transform(X_orig).to_numpy(), pre.transform(bg_rows).to_numpy()
    res["catatan_sampel"] = (f"{len(idx)} dari {len(test)} observasi uji dipilih acak ({int(y_sample.sum())} fraud); "
                             f"latar = {n_background} baris latih acak. Atribusi bersifat eksak terhadap latar ini.")

    shap_values, base_values, global_imp = {}, {}, {}
    # 2a. Regresi logistik: Shapley eksak bentuk tertutup (skala log-odds)
    coef = logit.params_.drop("const").to_numpy()
    phi, off = linear_shap(coef, Z_sample, Z_bg)
    shap_values["logit_mle"], base_values["logit_mle"] = phi, float(logit.params_["const"] + off)
    # 2b. Model pohon: Shapley eksak enumerasi (skala probabilitas)
    for name in TREE_MODELS:
        t = time.time()
        pipe, _ = load_model_bundle(name, MODELS_DIR)
        ex = ExactShapleyExplainer(model_scorer(pipe.named_steps["clf"]), Z_bg, selected)
        shap_values[name], base_values[name] = ex.shap_values(Z_sample), ex.expected_value
        _log(f"[2] Shapley {name}: {len(idx)} instance dalam {time.time() - t:.0f} detik")
    for name, phi in shap_values.items():
        global_imp[name] = mean_abs_shap(phi, selected)
    res["shap_global"] = {k: v for k, v in global_imp.items()}
    res["shap_nilai_dasar"] = base_values
    res["shap_satuan"] = {"logit_mle": "log-odds", "random_forest": "probabilitas", "hist_gb": "probabilitas"}
    _log("\n    Kepentingan global (porsi mean|SHAP|, %):")
    table_imp = pd.DataFrame({k: v.set_index("fitur")["porsi"] * 100 for k, v in global_imp.items()}).loc[selected]
    _log(table_imp.sort_values("logit_mle", ascending=False).round(1).to_string())

    # 3. Beeswarm dan dependence untuk 3 fitur teratas tiap model
    res["beeswarm"], res["dependence"] = {}, {}
    for name, phi in shap_values.items():
        res["beeswarm"][name] = beeswarm_data(phi, X_orig, 200, RANDOM_SEED)
        top3 = global_imp[name]["fitur"].head(3).tolist()
        res["dependence"][name] = {f: dependence_data(phi, X_orig, f) for f in top3}

    # 4. Permutation importance pada seluruh data uji (metrik PR-AUC)
    metric = lambda a, b: ranking_metrics(a, b)["pr_auc"]
    X_test = test[selected]
    predictors = {"logit_mle": lambda D: logit.predict_proba(pre.transform(D))[:, 1]}
    for name in TREE_MODELS:
        pipe, _ = load_model_bundle(name, MODELS_DIR)
        predictors[name] = (lambda p: (lambda D: p.predict_proba(D)[:, 1]))(pipe)
    perm = {name: permutation_importance(fn, X_test, yte, metric, n_repeats=10, seed=RANDOM_SEED)
            for name, fn in predictors.items()}
    res["permutation_importance"] = perm
    _log("\n[4] Permutation importance (penurunan PR-AUC uji), 5 teratas per model:")
    for name, df in perm.items():
        _log(f"    {name:14s} " + ", ".join(f"{r.fitur} {r.penurunan_mean:.3f}" for r in df.head(5).itertuples()))

    # 5. Kesepakatan peringkat antar model dan metode
    imps = {f"shap_{k}": v.set_index("fitur")["mean_abs_shap"] for k, v in global_imp.items()}
    imps.update({f"perm_{k}": v.set_index("fitur")["penurunan_mean"] for k, v in perm.items()})
    agree = rank_agreement(imps, top_k=3)
    res["kesepakatan_peringkat"] = agree
    _log("\n[5] Spearman antar peringkat kepentingan (SHAP vs SHAP antar model):")
    shap_names = [k for k in imps if k.startswith("shap_")]
    _log(agree["spearman"].loc[shap_names, shap_names].round(2).to_string())

    # 6. Penjelasan lokal: tiga observasi uji berisiko tertinggi menurut Random Forest
    rf_pipe, _ = load_model_bundle("random_forest", MODELS_DIR)
    p_rf = rf_pipe.predict_proba(X_test)[:, 1]
    top_idx = np.argsort(-p_rf)[:3]
    local = []
    for i in top_idx:
        row = X_test.iloc[i]
        z = pre.transform(X_test.iloc[[i]]).to_numpy()
        rf_ex = ExactShapleyExplainer(model_scorer(rf_pipe.named_steps["clf"]), Z_bg, selected)
        phi_rf = rf_ex.shap_values_one(z[0])
        phi_lg, off = linear_shap(coef, z, Z_bg)
        local.append({
            "company_id": test.loc[i, "company_id"], "tahun": int(test.loc[i, "tahun"]), "label_sebenarnya": int(yte[i]),
            "random_forest": explain_instance(phi_rf, row, rf_ex.expected_value, selected, train[selected], "probabilitas"),
            "regresi_logistik": explain_instance(phi_lg[0], row, float(logit.params_["const"] + off), selected, train[selected], "log-odds"),
            "kontribusi_linear_logodds": linear_contributions(logit.params_, pd.Series(z[0], index=selected), pd.Series(Z_bg.mean(axis=0), index=selected)),
        })
    res["penjelasan_lokal"] = local
    _log("\n[6] Penjelasan lokal (Random Forest) untuk 3 observasi uji berisiko tertinggi:")
    for e in local:
        _log(f"    {e['company_id']} tahun {e['tahun']} (label sebenarnya: {'FRAUD' if e['label_sebenarnya'] else 'non-fraud'}); "
             f"prediksi {e['random_forest']['prediksi']:.3f} (dasar {e['random_forest']['nilai_dasar']:.3f})")
        for s in e["random_forest"]["kalimat"][:3]:
            _log(f"       * {s}")

    res["perbandingan_interpretabilitas"] = pd.DataFrame([
        {"model": "Regresi Logistik", "tingkat": "tinggi", "bentuk_penjelasan": "koefisien, odds ratio dengan CI, uji Wald/LR; SHAP eksak bentuk tertutup",
         "catatan": "efek global konstan per fitur; mengasumsikan linearitas logit (diuji Box-Tidwell)"},
        {"model": "Random Forest", "tingkat": "sedang", "bentuk_penjelasan": "SHAP, permutation importance, dependence plot",
         "catatan": "menangkap nonlinearitas dan interaksi; tidak ada CI atau uji signifikansi bawaan"},
        {"model": "Gradient Boosting", "tingkat": "sedang", "bentuk_penjelasan": "SHAP, permutation importance, dependence plot",
         "catatan": "serupa Random Forest; lebih sensitif terhadap hiperparameter"},
        {"model": "Isolation Forest", "tingkat": "rendah", "bentuk_penjelasan": "skor anomali (tanpa atribusi per fitur di sini)",
         "catatan": "tak terawasi; skor bukan probabilitas"}])

    with open(PROCESSED_DIR / "explanation_results.json", "w", encoding="utf-8") as fh:
        json.dump(to_jsonable(res), fh, ensure_ascii=False, indent=2)
    _log(f"\nSelesai dalam {time.time() - t0:.0f} detik -> {PROCESSED_DIR / 'explanation_results.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-explain", type=int, default=200, help="jumlah observasi uji yang dijelaskan (acak)")
    ap.add_argument("--n-background", type=int, default=40, help="jumlah baris latar dari data latih")
    a = ap.parse_args()
    main(a.n_explain, a.n_background)
