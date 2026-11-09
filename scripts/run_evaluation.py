"""Tahap 5: evaluasi mendalam dan uji diagnostik.

    python -m scripts.run_evaluation                    # biaya FN:FP = 10:1, 1000 replikasi bootstrap
    python -m scripts.run_evaluation --c-fn 20 --n-boot 500

Prasyarat: run_feature_engineering dan run_modeling sudah dijalankan.
Keluaran: data/processed/evaluation_results.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from src.config import MODELS_DIR, PROCESSED_DIR, RANDOM_SEED
from src.data.schema import TARGET_COLUMN
from src.evaluation.calibration import IsotonicCalibrator, PlattCalibrator, calibration_report, calibration_table
from src.evaluation.diagnostics import (
    box_tidwell, hosmer_lemeshow, influence_diagnostics, influence_summary, residual_normality_note,
)
from src.evaluation.inference import (
    bootstrap_table, delong_auc_ci, delong_test, holm_correction, mcnemar_test, paired_cluster_bootstrap,
)
from src.evaluation.metrics import ranking_metrics, recall_precision_at_k
from src.evaluation.thresholds import (
    apply_threshold, bayes_threshold, cost_optimal_threshold, decision_curve, downsample_curve, max_f1_threshold,
    youden_threshold,
)
from src.features.selection import compute_vif
from src.features.transform import FeaturePreprocessor
from src.models.zoo import MODEL_SPECS
from src.utils import to_jsonable

MODELS = list(MODEL_SPECS)
PROB_MODELS = [m for m in MODELS if m != "isolation_forest"]      # skor Isolation Forest bukan probabilitas


def _log(msg: str) -> None:
    print(msg, flush=True)


def _load():
    needed = ["features.csv", "split.json", "feature_selection.json", "oof_predictions.csv",
              "test_predictions.csv", "modeling_results.json"]
    missing = [n for n in needed if not (PROCESSED_DIR / n).exists()]
    if missing:
        raise SystemExit(f"File belum ada: {', '.join(missing)}. Jalankan run_feature_engineering dan run_modeling dulu.")
    table = pd.read_csv(PROCESSED_DIR / "features.csv")
    split = json.loads((PROCESSED_DIR / "split.json").read_text(encoding="utf-8"))
    selected = json.loads((PROCESSED_DIR / "feature_selection.json").read_text(encoding="utf-8"))["selected"]
    oof = pd.read_csv(PROCESSED_DIR / "oof_predictions.csv")
    test_pred = pd.read_csv(PROCESSED_DIR / "test_predictions.csv")
    return table, split, selected, oof, test_pred


def main(c_fn: float = 10.0, c_fp: float = 1.0, n_boot: int = 1000) -> None:
    t0 = time.time()
    table, split, selected, oof, tp = _load()
    tr, te = np.array(split["train_idx"]), np.array(split["test_idx"])
    train, test = table.iloc[tr].reset_index(drop=True), table.iloc[te].reset_index(drop=True)
    ytr, yte = train[TARGET_COLUMN].to_numpy(int), test[TARGET_COLUMN].to_numpy(int)
    gte = test["company_id"].to_numpy()
    if not (np.array_equal(oof[TARGET_COLUMN], ytr) and np.array_equal(tp[TARGET_COLUMN], yte)):
        raise SystemExit("Label prediksi tidak cocok dengan split. Jalankan ulang run_modeling.")

    # Skor model logit (MLE) dan benchmark ditambahkan ke tabel skor
    logit_bundle = joblib.load(MODELS_DIR / "logit_mle.joblib")
    logit_model, pre = logit_bundle["model"], logit_bundle["preprocessor"]
    Ztr, Zte = pre.transform(train[selected]), pre.transform(test[selected])
    scores_test = {m: tp[m].to_numpy(float) for m in MODELS}
    scores_test["logit_mle"] = logit_model.predict_proba(Zte)[:, 1]
    scores_test["m_score"] = test["m_score"].fillna(train["m_score"].median()).to_numpy(float)
    oof_scores = {m: oof[m].to_numpy(float) for m in MODELS}
    res: dict = {"seed": RANDOM_SEED, "biaya": {"c_fn": c_fn, "c_fp": c_fp, "ambang_bayes": bayes_threshold(c_fn, c_fp)},
                 "n_uji": int(len(yte)), "fraud_uji": int(yte.sum()), "n_bootstrap": n_boot}
    _log(f"Uji: {len(yte)} baris ({yte.sum()} fraud, {len(np.unique(gte))} perusahaan). Biaya FN:FP = {c_fn:g}:{c_fp:g}")

    # A. Metrik uji dengan CI bootstrap per perusahaan
    metric_fns = {"roc_auc": lambda y, s: ranking_metrics(y, s)["roc_auc"],
                  "pr_auc": lambda y, s: ranking_metrics(y, s)["pr_auc"],
                  "recall_at_10": lambda y, s: recall_precision_at_k(y, s, 0.10)[0]}
    boot = []
    for name in MODELS + ["logit_mle", "m_score"]:
        t = bootstrap_table(yte, scores_test[name], gte, metric_fns, n_boot=n_boot, seed=RANDOM_SEED)
        t.insert(0, "model", name)
        boot.append(t)
    boot = pd.concat(boot, ignore_index=True)
    res["metrik_uji_bootstrap"] = boot
    delong_ci = {m: delong_auc_ci(yte, scores_test[m]) for m in MODELS + ["logit_mle", "m_score"]}
    res["auc_delong_ci"] = delong_ci
    _log("\n[A] Metrik uji dengan CI 95% (bootstrap per perusahaan):")
    for name in MODELS + ["logit_mle", "m_score"]:
        b = boot[boot.model == name].set_index("metrik")
        _log(f"    {name:17s} " + " | ".join(f"{k} {b.loc[k, 'estimasi']:.3f} [{b.loc[k, 'ci_bawah']:.3f}, {b.loc[k, 'ci_atas']:.3f}]"
                                               for k in ("roc_auc", "pr_auc", "recall_at_10")))

    # B. Kalibrasi: kalibrator dilatih pada OOF (latih), dievaluasi pada uji
    cal_rows, cal_curves, calibrated = [], {}, {}
    for m in PROB_MODELS + ["logit_mle"]:
        s_oof = oof_scores[m] if m in oof_scores else logit_model.predict_proba(Ztr)[:, 1]
        variants = {"mentah": scores_test[m]}
        variants["platt"] = PlattCalibrator().fit(s_oof, ytr).predict(scores_test[m])
        variants["isotonik"] = IsotonicCalibrator().fit(s_oof, ytr).predict(scores_test[m])
        calibrated[m] = variants["platt"]
        for v, p in variants.items():
            cal_rows.append({"model": m, "versi": v, **calibration_report(yte, p)})
            cal_curves[f"{m}:{v}"] = calibration_table(yte, p, n_bins=5)
    cal = pd.DataFrame(cal_rows)
    res["kalibrasi"], res["kurva_kalibrasi"] = cal, cal_curves
    _log("\n[B] Kalibrasi pada data uji (slope ideal 1, CITL ideal 0):")
    _log(cal[["model", "versi", "brier", "ece", "slope", "citl"]].round(3).to_string(index=False))

    # C. Ambang keputusan: dipilih pada OOF latih, diterapkan ke uji
    thr_rows = []
    for m in MODELS:
        s_oof, s_te = oof_scores[m], scores_test[m]
        rules = {"youden": youden_threshold(ytr, s_oof)["ambang"], "max_f1": max_f1_threshold(ytr, s_oof)["ambang"],
                 f"biaya_{c_fn:g}:{c_fp:g}": cost_optimal_threshold(ytr, s_oof, c_fn, c_fp)["ambang"]}
        for rule, t in rules.items():
            thr_rows.append({"model": m, "aturan": rule, **apply_threshold(yte, s_te, t, c_fn, c_fp)})
        if m in PROB_MODELS:      # aturan Bayes pada probabilitas yang sudah dikalibrasi (Platt)
            thr_rows.append({"model": m, "aturan": "bayes_platt",
                             **apply_threshold(yte, calibrated[m], bayes_threshold(c_fn, c_fp), c_fn, c_fp)})
    thr = pd.DataFrame(thr_rows)
    res["ambang_keputusan"] = thr
    _log("\n[C] Kinerja uji pada ambang yang dipilih dari data latih:")
    _log(thr[["model", "aturan", "threshold", "recall", "precision", "specificity", "f1", "persen_ditandai", "biaya_per_observasi", "biaya_tanpa_model_per_observasi", "penghematan_biaya_persen"]]
         .round(3).to_string(index=False))

    # D. Perbandingan model: DeLong (+Holm), bootstrap berpasangan PR-AUC, McNemar
    pairs = list(itertools.combinations(MODELS, 2))
    dl = []
    for a, b in pairs:
        r = delong_test(yte, scores_test[a], scores_test[b])
        pb = paired_cluster_bootstrap(yte, scores_test[a], scores_test[b], gte, metric_fns["pr_auc"],
                                      n_boot=min(n_boot, 500), seed=RANDOM_SEED)
        dl.append({"model_a": a, "model_b": b, "auc_a": r["auc_a"], "auc_b": r["auc_b"], "selisih_auc": r["selisih"],
                   "p_delong": r["p_value"], "selisih_pr_auc": pb["selisih"], "pr_ci_bawah": pb["ci_bawah"],
                   "pr_ci_atas": pb["ci_atas"], "p_bootstrap_pr": pb["p_value"]})
    dl = pd.DataFrame(dl)
    dl["p_delong_holm"] = holm_correction(dl["p_delong"])
    dl["p_bootstrap_pr_holm"] = holm_correction(dl["p_bootstrap_pr"])
    res["perbandingan_model"] = dl
    _log("\n[D] Perbandingan berpasangan (uji; koreksi Holm untuk 6 perbandingan):")
    _log(dl[["model_a", "model_b", "selisih_auc", "p_delong", "p_delong_holm", "selisih_pr_auc", "p_bootstrap_pr_holm"]].round(4).to_string(index=False))

    # McNemar pada kasus fraud saja dengan BEBAN KERJA SAMA (top 10% skor ditandai oleh setiap model).
    # McNemar pada seluruh baris didominasi true negative dan hanya mencerminkan perbedaan jumlah yang ditandai.
    flagged = {m: (scores_test[m] >= np.quantile(scores_test[m], 0.90)).astype(int) for m in MODELS}
    is_fraud = yte == 1
    mc = [{"model_a": a, "model_b": b, "n_fraud": int(is_fraud.sum()),
           **mcnemar_test(yte[is_fraud], flagged[a][is_fraud], flagged[b][is_fraud])} for a, b in pairs]
    res["mcnemar_sensitivitas_top10"] = pd.DataFrame(mc)
    _log("\n[D2] McNemar pada kasus fraud, beban kerja sama (top 10% ditandai); b = hanya A menangkap, c = hanya B menangkap:")
    _log(res["mcnemar_sensitivitas_top10"][["model_a", "model_b", "b", "c", "p_value", "metode"]].round(4).to_string(index=False))

    # E. Decision curve analysis (probabilitas terkalibrasi Platt)
    dca = {m: decision_curve(yte, calibrated[m], np.linspace(0.02, 0.30, 15)) for m in PROB_MODELS}
    res["decision_curve"] = dca
    useful = {m: float((dca[m]["net_benefit_model"] > np.maximum(dca[m]["net_benefit_tandai_semua"], 0)).mean()) for m in PROB_MODELS}
    res["dca_proporsi_ambang_berguna"] = useful
    _log(f"\n[E] Decision curve: proporsi ambang 2-30% di mana model > (tandai semua, tidak tandai): {useful}")

    # F. Titik kurva ROC / PR
    curves = {}
    for m in MODELS + ["logit_mle"]:
        fpr, tpr, _ = roc_curve(yte, scores_test[m])
        prec, rec, _ = precision_recall_curve(yte, scores_test[m])
        curves[m] = {"roc": dict(zip(("fpr", "tpr"), downsample_curve(fpr, tpr))),
                     "pr": dict(zip(("recall", "precision"), downsample_curve(rec, prec)))}
    res["kurva"] = curves

    # G. Diagnostik regresi logistik (model inferensial pada fitur terstandardisasi)
    ptr, pte = logit_model.predict_proba(Ztr)[:, 1], scores_test["logit_mle"]
    hl_tr, hl_te = hosmer_lemeshow(ytr, ptr), hosmer_lemeshow(yte, pte, groups=5)
    pre_raw = FeaturePreprocessor(scale=False).fit(train[selected])
    bt = box_tidwell(pre_raw.transform(train[selected]), ytr)
    diag = influence_diagnostics(logit_model, Ztr, ytr)
    summ = influence_summary(diag, logit_model.n_params_)
    top = diag.assign(company_id=train["company_id"], tahun=train["tahun"], y=ytr).sort_values("cooks_d", ascending=False).head(10)
    vif = compute_vif(Ztr)
    res["diagnostik_logit"] = {
        "hosmer_lemeshow_latih": {k: hl_tr[k] for k in ("statistik", "dof", "p_value")}, "tabel_hl_latih": hl_tr["tabel"],
        "hosmer_lemeshow_uji": {k: hl_te[k] for k in ("statistik", "dof", "p_value")},
        "box_tidwell": bt, "pengaruh_ringkasan": summ, "pengaruh_teratas": top.reset_index(drop=True),
        "vif": vif, "catatan_residual": residual_normality_note(), "kecocokan_model": logit_model.model_fit()}
    _log(f"\n[G] Regresi logistik: Hosmer-Lemeshow latih chi2={hl_tr['statistik']:.2f} (dof={hl_tr['dof']}) p={hl_tr['p_value']:.3f}; "
         f"uji p={hl_te['p_value']:.3f}")
    _log("    Box-Tidwell (linearitas logit; ambang Bonferroni %.4f):\n%s" % (
        bt["ambang_bonferroni"].iloc[0], bt[["fitur", "p_value", "linear"]].round(4).to_string(index=False)))
    _log(f"    Pengaruh: Cook's D > 4/n: {summ['cooks_d_gt_4n']} | Cook's D maks {summ['cooks_d_maks']:.3f} | "
         f"|dev|>2: {summ['persen_resid_deviance_gt_2']:.1f}% | VIF maks {vif.max():.2f}")

    (PROCESSED_DIR / "evaluation_results.json").write_text(
        json.dumps(to_jsonable(res), ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"\nSelesai dalam {time.time() - t0:.0f} detik -> {PROCESSED_DIR / 'evaluation_results.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--c-fn", type=float, default=10.0, help="biaya false negative (fraud lolos)")
    ap.add_argument("--c-fp", type=float, default=1.0, help="biaya false positive (audit sia-sia)")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    main(a.c_fn, a.c_fp, a.n_boot)
