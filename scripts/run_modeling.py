"""Tahap 4: pemodelan lengkap.

    python -m scripts.run_modeling                    # lengkap (sekitar 5 menit)
    python -m scripts.run_modeling --quick            # nested CV 3 lipatan, tanpa ablasi fitur
    python -m scripts.run_modeling --skip-ablation    # tanpa ablasi fitur (lebih cepat)
    python -m scripts.run_modeling --ablation-only    # hanya ablasi; menambahkan hasil ke modeling_results.json

Prasyarat: ``python -m scripts.run_feature_engineering`` sudah dijalankan.
Keluaran di data/processed/ dan models/.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import MODELS_DIR, PROCESSED_DIR, RANDOM_SEED
from src.data.schema import TARGET_COLUMN
from src.data.split import split_summary, temporal_split
from src.evaluation.metrics import ranking_metrics
from src.features.ratios import FEATURE_NAMES
from src.features.transform import FeaturePreprocessor
from src.models.logit import LogitInference, add_unit_odds_ratios
from src.models.training import (
    class_distribution_after, compare_imbalance_strategies, cross_validate_model, evaluate_on_holdout,
    library_versions, nested_cv, save_model_bundle, tune_model,
)
from src.models.zoo import MODEL_SPECS, make_model
from src.utils import to_jsonable

SEED = RANDOM_SEED


def _log(msg: str) -> None:
    print(msg, flush=True)


def _load_inputs() -> tuple[pd.DataFrame, dict, list[str]]:
    needed = [PROCESSED_DIR / n for n in ("features.csv", "split.json", "feature_selection.json")]
    missing = [p.name for p in needed if not p.exists()]
    if missing:
        raise SystemExit(f"File belum ada: {', '.join(missing)}. Jalankan dulu: python -m scripts.run_feature_engineering")
    table = pd.read_csv(needed[0])
    split = json.loads(needed[1].read_text(encoding="utf-8"))
    selected = json.loads(needed[2].read_text(encoding="utf-8"))["selected"]
    if len(split["train_idx"]) + len(split["test_idx"]) != len(table):
        raise SystemExit("split.json tidak cocok dengan features.csv. Jalankan ulang run_feature_engineering.")
    return table, split, selected


def _best_strategy(comp: pd.DataFrame, model: str) -> str:
    sub = comp[comp.model == model]
    return str(sub.loc[sub["pr_auc_mean"].idxmax(), "strategi"])


def _ablation_only(train: pd.DataFrame, ytr: np.ndarray, gtr: np.ndarray, selected: list[str]) -> None:
    """Ablasi seleksi fitur memakai strategi dan parameter dari modeling_results.json yang sudah ada."""
    path = PROCESSED_DIR / "modeling_results.json"
    if not path.exists():
        raise SystemExit("modeling_results.json belum ada. Jalankan dulu: python -m scripts.run_modeling --skip-ablation")
    results = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for name in ("logreg", "random_forest"):
        cfg = results["nested_cv"][name]
        for label, cols in (("terpilih", selected), ("semua_fitur", FEATURE_NAMES)):
            r = cross_validate_model(train[cols], ytr, gtr, name, cfg["strategi"], cfg["parameter_final"], seed=SEED).summary()
            rows.append({"model": name, "himpunan_fitur": label, "n_fitur": len(cols), **r})
    df = pd.DataFrame(rows)
    results["ablasi_fitur"] = to_jsonable(df)
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    _log("Ablasi seleksi fitur (CV, data latih):\n" + df[["model", "himpunan_fitur", "n_fitur", "pr_auc_mean", "pr_auc_sd", "roc_auc_mean"]].round(3).to_string(index=False))


def main(quick: bool = False, skip_ablation: bool = False, ablation_only: bool = False) -> None:
    t0 = time.time()
    table, split, selected = _load_inputs()
    tr, te = np.array(split["train_idx"]), np.array(split["test_idx"])
    train, test = table.iloc[tr].reset_index(drop=True), table.iloc[te].reset_index(drop=True)
    ytr, yte = train[TARGET_COLUMN].to_numpy(int), test[TARGET_COLUMN].to_numpy(int)
    gtr = train["company_id"].to_numpy()
    Xtr, Xte = train[selected], test[selected]
    if ablation_only:
        _ablation_only(train, ytr, gtr, selected)
        return
    results: dict = {"seed": SEED, "fitur_terpilih": selected, "split": split["ringkasan"], "versi_library": library_versions()}
    _log(f"Data latih {len(train)} baris ({ytr.sum()} fraud) | uji {len(test)} baris ({yte.sum()} fraud) | fitur: {len(selected)}")

    # 1. Distribusi kelas sebelum/sesudah
    dist = class_distribution_after(ytr, seed=SEED)
    results["distribusi_kelas_resampling"] = dist
    _log("\n[1] Distribusi kelas pada data latih:\n" + dist.drop(columns="catatan").to_string(index=False))

    # 2. Strategi imbalance (sebelum vs sesudah) pada dua model utama
    comp = compare_imbalance_strategies(Xtr, ytr, gtr, models=("logreg", "random_forest"), seed=SEED)
    results["perbandingan_imbalance"] = comp
    chosen = {m: _best_strategy(comp, m) for m in ("logreg", "random_forest")}
    chosen["hist_gb"] = chosen["random_forest"]          # model berbasis pohon memakai strategi yang sama
    chosen["isolation_forest"] = "none"
    _log("\n[2] Perbandingan strategi imbalance (CV 5-lipatan per perusahaan, data latih):")
    _log(comp[["model", "strategi", "pr_auc_mean", "pr_auc_sd", "roc_auc_mean", "recall_at_10_mean"]].round(3).to_string(index=False))
    _log(f"    Strategi terpilih (PR-AUC CV tertinggi): {chosen}")

    # 3. Tuning + nested CV
    final_params, nested_summary, tuning_tables = {}, {}, {}
    n_outer = 3 if quick else 5
    for name in MODEL_SPECS:
        strat = chosen[name]
        t = time.time()
        nres = nested_cv(Xtr, ytr, gtr, name, strat, n_outer=n_outer, n_inner=3, seed=SEED)
        best, tab = tune_model(Xtr, ytr, gtr, name, strat, n_splits=3, seed=SEED)
        final_params[name], tuning_tables[name] = best, tab
        nested_summary[name] = {"strategi": strat, "parameter_final": best,
                                "pr_auc_mean": float(nres["pr_auc"].mean()), "pr_auc_sd": float(nres["pr_auc"].std(ddof=1)),
                                "roc_auc_mean": float(nres["roc_auc"].mean()), "recall_at_10_mean": float(nres["recall_at_10"].mean()),
                                "parameter_per_lipatan": nres["parameter_terpilih"].tolist()}
        _log(f"[3] {name:17s} nested CV PR-AUC {nested_summary[name]['pr_auc_mean']:.3f}±{nested_summary[name]['pr_auc_sd']:.3f}"
             f" | params {best} | {time.time() - t:.0f}s")
    results["nested_cv"], results["tuning"] = nested_summary, tuning_tables

    # 4. Ablasi seleksi fitur (semua 36 fitur vs terpilih)
    if not (quick or skip_ablation):
        ablation = []
        for name in ("logreg", "random_forest"):
            for label, cols in (("terpilih", selected), ("semua_fitur", FEATURE_NAMES)):
                r = cross_validate_model(train[cols], ytr, gtr, name, chosen[name], final_params[name], seed=SEED).summary()
                ablation.append({"model": name, "himpunan_fitur": label, "n_fitur": len(cols), **r})
        results["ablasi_fitur"] = pd.DataFrame(ablation)
        _log("\n[4] Ablasi seleksi fitur (CV):\n" + results["ablasi_fitur"][["model", "himpunan_fitur", "n_fitur", "pr_auc_mean", "roc_auc_mean"]].round(3).to_string(index=False))

    # 5. Model final: OOF (untuk tahap evaluasi), hold-out, temporal
    oof = pd.DataFrame({"company_id": train["company_id"], "tahun": train["tahun"], TARGET_COLUMN: ytr})
    rows_test, test_scores = [], pd.DataFrame({"company_id": test["company_id"], "tahun": test["tahun"], TARGET_COLUMN: yte})
    t_tr, t_te = temporal_split(table, 1)
    temporal_rows = []
    for name, spec in MODEL_SPECS.items():
        strat, params = chosen[name], final_params[name]
        oof[name] = cross_validate_model(Xtr, ytr, gtr, name, strat, params, seed=SEED).oof_scores
        pipe, m, scores = evaluate_on_holdout(Xtr, ytr, Xte, yte, name, strat, params, SEED)
        rows_test.append({"model": name, "label": spec.label, "strategi": strat, **m})
        test_scores[name] = scores
        save_model_bundle(pipe, name, MODELS_DIR, {"fitur": selected, "strategi_imbalance": strat, "parameter": params,
                                                    "seed": SEED, "metrik_uji": m, "metrik_nested_cv": nested_summary[name]})
        # robustness temporal: latih tahun awal, uji tahun terakhir (konfigurasi sama, tanpa tuning ulang)
        tp = make_model(name, params, strat, SEED).fit(table.iloc[t_tr][selected], table.iloc[t_tr][TARGET_COLUMN].to_numpy(int))
        ts = tp.predict_proba(table.iloc[t_te][selected])[:, 1]
        temporal_rows.append({"model": name, **ranking_metrics(table.iloc[t_te][TARGET_COLUMN].to_numpy(int), ts)})

    # Benchmark berbasis teori (tanpa pelatihan): Beneish M-Score dan Altman Z''
    for label, col, sign in (("Beneish M-Score (benchmark)", "m_score", 1), ("Altman Z'' (benchmark, dibalik)", "altman_z2", -1)):
        s = (sign * test[col]).fillna(float((sign * train[col]).median()))
        rows_test.append({"model": col, "label": label, "strategi": "-", **ranking_metrics(yte, s.to_numpy())})
    results["uji_holdout"] = pd.DataFrame(rows_test)
    results["uji_temporal"] = pd.DataFrame(temporal_rows)
    results["split_temporal"] = split_summary(table, t_tr, t_te)
    _log("\n[5] Kinerja pada DATA UJI hold-out (perusahaan tidak pernah terlihat):")
    _log(results["uji_holdout"][["label", "strategi", "pr_auc", "roc_auc", "recall_at_10", "precision_at_10"]].round(3).to_string(index=False))
    _log("    Uji temporal (latih <= 2022, uji 2023):")
    _log(results["uji_temporal"].round(3).to_string(index=False))

    # 6. Regresi logistik inferensial (MLE dan Firth) pada fitur terstandardisasi
    pre = FeaturePreprocessor().fit(Xtr)
    Ztr, Zte = pre.transform(Xtr), pre.transform(Xte)
    logit_out = {}
    for label, firth in (("mle", False), ("firth", True)):
        m = LogitInference(firth=firth).fit(Ztr, ytr)
        summ = add_unit_odds_ratios(m.summary(), pre.std_).reset_index()
        pt = m.predict_proba(Zte)[:, 1]
        logit_out[label] = {"ringkasan": summ, "kecocokan_model": m.model_fit(), "uji": ranking_metrics(yte, pt)}
        save_model_bundle({"model": m, "preprocessor": pre}, f"logit_{label}", MODELS_DIR,
                          {"fitur": selected, "seed": SEED, "firth": firth, "kecocokan_model": m.model_fit(),
                           "catatan": "Koefisien pada fitur terstandardisasi (odds ratio per 1 SD)."})
    results["logit_inferensial"] = logit_out
    _log("\n[6] Regresi logistik (MLE), odds ratio per 1 SD dengan CI 95%:")
    s = logit_out["mle"]["ringkasan"]
    s = s[s["variabel"] != "const"]
    _log(s[["variabel", "odds_ratio", "or_ci_bawah", "or_ci_atas", "p_value"]].round(4).to_string(index=False))
    fit = logit_out["mle"]["kecocokan_model"]
    _log(f"    LR chi2={fit['lr_chi2']:.1f} (df={fit['lr_df']}), p={fit['lr_p_value']:.2e}, McFadden R2={fit['pseudo_r2_mcfadden']:.3f}, "
         f"AIC={fit['aic']:.1f}, konvergen={fit['konvergen']}, separasi={fit['separasi_dicurigai']}")

    # 7. Simpan
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    oof.to_csv(PROCESSED_DIR / "oof_predictions.csv", index=False)
    test_scores.to_csv(PROCESSED_DIR / "test_predictions.csv", index=False)
    Path(PROCESSED_DIR / "modeling_results.json").write_text(
        json.dumps(to_jsonable(results), ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"\nSelesai dalam {time.time() - t0:.0f} detik. Hasil: {PROCESSED_DIR}, model: {MODELS_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="nested CV 3 lipatan dan tanpa ablasi fitur")
    ap.add_argument("--skip-ablation", action="store_true", help="lewati ablasi seleksi fitur")
    ap.add_argument("--ablation-only", action="store_true", help="jalankan ablasi saja (butuh modeling_results.json)")
    a = ap.parse_args()
    main(quick=a.quick, skip_ablation=a.skip_ablation, ablation_only=a.ablation_only)
