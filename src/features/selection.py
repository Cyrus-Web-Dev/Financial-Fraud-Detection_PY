"""Seleksi fitur berbasis statistik (HANYA dijalankan pada data latih).

Tahapan ``run_feature_selection``
---------------------------------
1. Preprocessing (winsorize, imputasi median, standardisasi) + pembuangan kolom
   kosong/konstan.
2. Skrining univariat (AUC Mann-Whitney, q-value BH). Bersifat informatif dan
   dipakai sebagai pemutus seri; fitur TIDAK dibuang berdasarkan p-value univariat
   karena menyaring dengan uji univariat mengabaikan efek gabungan dan menggelembungkan
   galat tipe I (Harrell, 2015).
3. Filter korelasi: pasangan |rho Spearman| > ambang -> buang yang AUC-nya lebih lemah.
4. VIF iteratif: buang fitur ber-VIF tertinggi sampai semua VIF <= ambang.
5. Stability selection Lasso (Meinshausen & Buhlmann, 2010): L1-logistic dijalankan
   berulang pada subsampel PERUSAHAAN (bukan baris, agar struktur panel terjaga);
   fitur dipilih bila frekuensi terpilih >= ambang.

Fitur "terlindungi" (indeks Beneish, berbasis teori) tidak dibuang oleh filter
otomatis demi interpretabilitas; bila VIF-nya tetap tinggi, hal itu dilaporkan.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.data.schema import TARGET_COLUMN
from src.data.split import group_stratified_cv
from src.evaluation.eda import compare_groups
from src.features.ratios import BENEISH_INDEX_NAMES, FEATURE_CATALOG
from src.features.transform import FeaturePreprocessor
from src.models.compat import make_logreg
from src.utils import to_jsonable

VIF_CAP: float = 1e6  # nilai VIF yang dianggap "tak hingga" (kolinearitas sempurna)


# ---------------------------------------------------------------------------
# VIF
# ---------------------------------------------------------------------------
def compute_vif(X: pd.DataFrame) -> pd.Series:
    """Variance Inflation Factor: VIF_j = 1 / (1 - R^2_j), R^2_j dari regresi kolom j pada
    kolom lain (dengan intersep). Identik dengan ``statsmodels.variance_inflation_factor``.

    Kolinearitas sempurna menghasilkan ``VIF_CAP`` (dilaporkan sebagai sangat besar).
    """
    if X.shape[1] < 2:
        raise ValueError("VIF membutuhkan minimal 2 kolom.")
    A = X.to_numpy(dtype=float)
    if not np.isfinite(A).all():
        raise ValueError("VIF tidak dapat dihitung: terdapat NaN/inf. Lakukan imputasi lebih dulu.")
    n, p = A.shape
    out = np.empty(p)
    for j in range(p):
        yj = A[:, j]
        others = np.delete(A, j, axis=1)
        design = np.column_stack([np.ones(n), others])
        beta, *_ = np.linalg.lstsq(design, yj, rcond=None)
        resid = yj - design @ beta
        ss_tot = float(((yj - yj.mean()) ** 2).sum())
        if ss_tot <= 0:
            out[j] = np.nan
            continue
        r2 = 1.0 - float((resid ** 2).sum()) / ss_tot
        out[j] = VIF_CAP if (1.0 - r2) <= 1.0 / VIF_CAP else min(1.0 / (1.0 - r2), VIF_CAP)
    return pd.Series(out, index=X.columns, name="vif")


def iterative_vif_pruning(X: pd.DataFrame, threshold: float = 10.0, protected: Sequence[str] = ()
                          ) -> tuple[list[str], pd.DataFrame, pd.Series]:
    """Buang fitur ber-VIF tertinggi satu per satu sampai semua VIF <= ``threshold``.

    Fitur ``protected`` tidak dibuang; bila hanya fitur terlindungi yang masih melewati
    ambang, proses berhenti dan kondisi itu terlihat pada VIF akhir.

    Returns: (fitur_tersisa, log_pembuangan[langkah, fitur, vif], vif_akhir)
    """
    protected = set(protected)
    cols = list(X.columns)
    log: list[dict] = []
    while len(cols) > 1:
        vif = compute_vif(X[cols])
        over = vif[(vif > threshold) | vif.isna()]
        droppable = over[[c for c in over.index if c not in protected]]
        if droppable.empty:
            break
        worst = str(droppable.fillna(np.inf).idxmax())
        log.append({"langkah": len(log) + 1, "fitur": worst, "vif": float(vif[worst])})
        cols.remove(worst)
    final_vif = compute_vif(X[cols]) if len(cols) > 1 else pd.Series(1.0, index=cols, name="vif")
    return cols, pd.DataFrame(log, columns=["langkah", "fitur", "vif"]), final_vif


# ---------------------------------------------------------------------------
# Skrining univariat dan filter korelasi
# ---------------------------------------------------------------------------
def univariate_screening(X: pd.DataFrame, y: np.ndarray | pd.Series) -> pd.DataFrame:
    """AUC univariat (P(fraud > non-fraud) + 0.5 P(sama)), p-value Mann-Whitney, q-value BH.

    ``auc_kuat = max(AUC, 1-AUC)`` mengukur kekuatan diskriminasi tanpa memandang arah.
    """
    frame = X.copy()
    frame[TARGET_COLUMN] = np.asarray(y)
    res = compare_groups(frame, list(X.columns), TARGET_COLUMN)
    res["auc"] = (res["efek_rank_biserial"] + 1.0) / 2.0
    res["auc_kuat"] = np.maximum(res["auc"], 1.0 - res["auc"])
    res["arah"] = np.where(res["auc"] >= 0.5, "+", "-")
    return res[["variabel", "auc", "auc_kuat", "arah", "p_value", "q_value"]].rename(columns={"variabel": "fitur"})


def correlation_filter(X: pd.DataFrame, strength: pd.Series, threshold: float = 0.9,
                       protected: Sequence[str] = ()) -> tuple[list[str], pd.DataFrame]:
    """Untuk setiap pasangan |Spearman| > threshold, buang fitur dengan ``strength`` lebih rendah
    (kecuali terlindungi). ``strength`` diindeks nama fitur."""
    protected = set(protected)
    corr = X.corr(method="spearman")
    cols = list(X.columns)
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if np.isfinite(r) and abs(r) > threshold:
                pairs.append((abs(r), cols[i], cols[j]))
    pairs.sort(reverse=True)
    dropped: dict[str, dict] = {}
    for r, a, b in pairs:
        if a in dropped or b in dropped:
            continue
        cand = [c for c in (a, b) if c not in protected]
        if not cand:
            continue
        loser = min(cand, key=lambda c: strength.get(c, 0.0))
        keeper = b if loser == a else a
        dropped[loser] = {"fitur": loser, "berkorelasi_dengan": keeper, "rho": float(r)}
    kept = [c for c in cols if c not in dropped]
    return kept, pd.DataFrame(list(dropped.values()), columns=["fitur", "berkorelasi_dengan", "rho"])


# ---------------------------------------------------------------------------
# Lasso + stability selection
# ---------------------------------------------------------------------------
def choose_l1_c(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, grid: Sequence[float] | None = None,
                n_splits: int = 5, seed: int = 42) -> tuple[float, pd.DataFrame]:
    """Pilih C (penalti L1) dengan CV stratified-group; skor = PR-AUC (average precision)."""
    grid = list(grid) if grid is not None else list(np.logspace(-3, 0, 10))
    folds = list(group_stratified_cv(y, groups, n_splits=n_splits, seed=seed))
    A = X.to_numpy(dtype=float)
    rows = []
    for C in grid:
        scores = []
        for tr, te in folds:
            m = make_logreg("l1", C=C, class_weight="balanced", random_state=seed).fit(A[tr], y[tr])
            scores.append(average_precision_score(y[te], m.decision_function(A[te])))
        rows.append({"C": float(C), "pr_auc_mean": float(np.mean(scores)), "pr_auc_sd": float(np.std(scores, ddof=1))})
    table = pd.DataFrame(rows)
    # aturan 1-SE: C terkecil (model paling sederhana) yang skornya >= terbaik - 1 SE
    best = table["pr_auc_mean"].idxmax()
    cutoff = table.loc[best, "pr_auc_mean"] - table.loc[best, "pr_auc_sd"] / np.sqrt(len(folds))
    chosen = float(table.loc[table["pr_auc_mean"] >= cutoff, "C"].min())
    return chosen, table


def stability_selection(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, C: float, n_boot: int = 100,
                        sample_frac: float = 0.7, seed: int = 42) -> pd.Series:
    """Frekuensi (0-1) tiap fitur bernilai koefisien L1 != 0 pada subsampel perusahaan acak."""
    rng = np.random.default_rng(seed)
    A = X.to_numpy(dtype=float)
    uniq = np.unique(groups)
    k = max(2, int(round(sample_frac * len(uniq))))
    counts = np.zeros(A.shape[1])
    valid = 0
    for _ in range(n_boot):
        chosen = rng.choice(uniq, size=k, replace=False)
        mask = np.isin(groups, chosen)
        ys = y[mask]
        if ys.sum() < 5 or (len(ys) - ys.sum()) < 5:
            continue
        m = make_logreg("l1", C=C, class_weight="balanced", random_state=seed).fit(A[mask], ys)
        counts += (m.coef_.ravel() != 0)
        valid += 1
    if valid == 0:
        raise ValueError("Stability selection gagal: tidak ada subsampel dengan cukup kejadian fraud.")
    return pd.Series(counts / valid, index=X.columns, name="frekuensi_terpilih")


# ---------------------------------------------------------------------------
# Orkestrasi
# ---------------------------------------------------------------------------
@dataclass
class FeatureSelectionResult:
    selected: list[str]
    preprocessor: FeaturePreprocessor
    report: pd.DataFrame                      # satu baris per fitur awal
    steps: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return to_jsonable({"selected": self.selected, "report": self.report, "steps": self.steps})


def run_feature_selection(
    X_train: pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    groups_train: np.ndarray | pd.Series,
    protected: Sequence[str] = tuple(BENEISH_INDEX_NAMES),
    corr_threshold: float = 0.90,
    vif_threshold: float = 10.0,
    stability_threshold: float = 0.60,
    n_boot: int = 100,
    seed: int = 42,
) -> FeatureSelectionResult:
    """Jalankan seluruh pipeline seleksi pada DATA LATIH dan kembalikan fitur terpilih + laporan."""
    y = np.asarray(y_train).astype(int)
    groups = np.asarray(groups_train)
    if len(X_train) != len(y) or len(y) != len(groups):
        raise ValueError("X_train, y_train, dan groups_train harus sama panjang.")
    if y.sum() < 10:
        raise ValueError("Kejadian fraud pada data latih terlalu sedikit (< 10) untuk seleksi fitur.")

    labels = {f.name: f.label for f in FEATURE_CATALOG}
    groups_of = {f.name: f.group for f in FEATURE_CATALOG}
    all_features = list(X_train.columns)
    status: dict[str, tuple[str, str]] = {}  # fitur -> (status, alasan)

    # 1. Preprocessing
    pre = FeaturePreprocessor().fit(X_train)
    Xp = pre.transform(X_train)
    for col, why in pre.dropped_.items():
        status[col] = ("dibuang", f"preprocessing: {why}")
    current = list(Xp.columns)
    protected_now = [c for c in protected if c in current]

    # 2. Skrining univariat
    screen = univariate_screening(Xp, y).set_index("fitur")

    # 3. Filter korelasi
    current, corr_drops = correlation_filter(Xp[current], screen["auc_kuat"], corr_threshold, protected_now)
    for _, r in corr_drops.iterrows():
        status[r["fitur"]] = ("dibuang", f"korelasi {r['rho']:.2f} dengan '{r['berkorelasi_dengan']}'")

    # 4. VIF iteratif
    current, vif_log, vif_final = iterative_vif_pruning(Xp[current], vif_threshold, protected_now)
    for _, r in vif_log.iterrows():
        status[r["fitur"]] = ("dibuang", f"VIF {r['vif']:.1f} > {vif_threshold:g}")

    # 5. Lasso + stability selection
    C, c_table = choose_l1_c(Xp[current], y, groups, seed=seed)
    freq = stability_selection(Xp[current], y, groups, C, n_boot=n_boot, seed=seed)
    selected: list[str] = []
    for col in current:
        stable = freq[col] >= stability_threshold
        if stable:
            selected.append(col)
            status[col] = ("dipertahankan", f"stabil pada Lasso (frekuensi {freq[col]:.0%})")
        elif col in protected_now:
            selected.append(col)
            status[col] = ("dipertahankan", f"indeks Beneish (domain); frekuensi Lasso {freq[col]:.0%}")
        else:
            status[col] = ("dibuang", f"tidak stabil pada Lasso (frekuensi {freq[col]:.0%})")

    rows = []
    for col in all_features:
        st, why = status.get(col, ("dibuang", "tidak diketahui"))
        rows.append({
            "fitur": col, "label": labels.get(col, col), "kelompok": groups_of.get(col, "-"),
            "status": st, "alasan": why,
            "auc_univariat": float(screen.loc[col, "auc"]) if col in screen.index else np.nan,
            "q_value": float(screen.loc[col, "q_value"]) if col in screen.index else np.nan,
            "vif_akhir": float(vif_final[col]) if col in vif_final.index else np.nan,
            "frekuensi_lasso": float(freq[col]) if col in freq.index else np.nan,
        })
    report = pd.DataFrame(rows)

    # Pre-processor final hanya untuk fitur terpilih (agar transform data baru cukup memakai fitur itu)
    final_pre = FeaturePreprocessor().fit(X_train[selected])
    protected_over = [c for c in protected_now if c in vif_final.index and vif_final[c] > vif_threshold]

    return FeatureSelectionResult(
        selected=selected, preprocessor=final_pre, report=report,
        steps={
            "pembuangan_preprocessing": pre.dropped_,
            "pembuangan_korelasi": corr_drops,
            "pembuangan_vif": vif_log,
            "vif_akhir": vif_final.sort_values(ascending=False),
            "pemilihan_C_lasso": {"C_terpilih": C, "tabel_cv": c_table},
            "fitur_terlindungi_vif_tinggi": protected_over,
            "parameter": {"corr_threshold": corr_threshold, "vif_threshold": vif_threshold,
                          "stability_threshold": stability_threshold, "n_boot": n_boot, "seed": seed},
        },
    )
