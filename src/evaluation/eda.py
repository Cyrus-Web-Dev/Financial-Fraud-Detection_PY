"""Analisis data eksploratif (EDA) berbasis statistik.

Semua fungsi menerima DataFrame dan mengembalikan DataFrame/dict sederhana
sehingga mudah ditampilkan di API dan frontend. Fungsi ``compare_groups`` dan
``correlation_matrix`` juga dipakai ulang pada tahap rasio keuangan.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from src.data.schema import TARGET_COLUMN

SHAPIRO_MAX_N: int = 5000  # batas praktis uji Shapiro-Wilk


def numeric_columns(df: pd.DataFrame, exclude: Sequence[str] = ()) -> list[str]:
    """Kolom numerik (selain yang dikecualikan) pada ``df``."""
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def descriptive_statistics(df: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Statistik deskriptif lengkap: n, missing, mean, sd, kuartil, skewness, kurtosis.

    Kurtosis dilaporkan sebagai *excess kurtosis* (normal = 0).
    """
    cols = list(columns) if columns is not None else numeric_columns(df)
    rows = []
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce").astype(float)
        v = x.dropna().to_numpy()
        n = int(v.size)
        row = {"kolom": c, "n": n, "missing": int(x.isna().sum()),
               "persen_missing": float(x.isna().mean() * 100.0)}
        if n == 0:
            row.update({k: np.nan for k in ("mean", "sd", "min", "q1", "median", "q3", "max", "skewness", "kurtosis")})
        else:
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            row.update({
                "mean": float(v.mean()),
                "sd": float(v.std(ddof=1)) if n > 1 else np.nan,
                "min": float(v.min()), "q1": float(q1), "median": float(med), "q3": float(q3),
                "max": float(v.max()),
                "skewness": float(stats.skew(v, bias=False)) if n > 2 else np.nan,
                "kurtosis": float(stats.kurtosis(v, bias=False)) if n > 3 else np.nan,
            })
        rows.append(row)
    return pd.DataFrame(rows)


def normality_tests(df: pd.DataFrame, columns: Sequence[str] | None = None,
                    seed: int = 42) -> pd.DataFrame:
    """Uji normalitas Shapiro-Wilk (sampel acak maksimal 5000) dan D'Agostino-Pearson.

    Pada data keuangan hampir semua pos akan menolak normalitas; hasil ini
    menjadi dasar memakai uji nonparametrik dan transformasi (log/winsorize).
    """
    rng = np.random.default_rng(seed)
    cols = list(columns) if columns is not None else numeric_columns(df)
    rows = []
    for c in cols:
        v = pd.to_numeric(df[c], errors="coerce").dropna().to_numpy(dtype=float)
        row: dict[str, float | str | int] = {"kolom": c, "n": int(v.size),
                                             "shapiro_W": np.nan, "shapiro_p": np.nan,
                                             "dagostino_p": np.nan, "normal": "tidak dapat diuji"}
        if v.size >= 8 and np.ptp(v) > 0:
            sample = v if v.size <= SHAPIRO_MAX_N else rng.choice(v, SHAPIRO_MAX_N, replace=False)
            w, p = stats.shapiro(sample)
            row["shapiro_W"], row["shapiro_p"] = float(w), float(p)
            row["dagostino_p"] = float(stats.normaltest(v).pvalue)
            row["normal"] = "ya" if p >= 0.05 else "tidak"
        rows.append(row)
    return pd.DataFrame(rows)


def correlation_matrix(df: pd.DataFrame, columns: Sequence[str] | None = None,
                       method: str = "spearman") -> pd.DataFrame:
    """Matriks korelasi (``spearman`` default, tahan outlier dan relasi monoton)."""
    if method not in {"pearson", "spearman", "kendall"}:
        raise ValueError("method harus 'pearson', 'spearman', atau 'kendall'.")
    cols = list(columns) if columns is not None else numeric_columns(df, exclude=[TARGET_COLUMN])
    return df[cols].apply(pd.to_numeric, errors="coerce").corr(method=method)


def high_correlation_pairs(corr: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """Pasangan variabel dengan |korelasi| >= ``threshold`` (segitiga atas saja)."""
    cols = list(corr.columns)
    rows = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if np.isfinite(r) and abs(r) >= threshold:
                rows.append({"variabel_1": cols[i], "variabel_2": cols[j], "korelasi": float(r)})
    out = pd.DataFrame(rows, columns=["variabel_1", "variabel_2", "korelasi"])
    return out.reindex(out["korelasi"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def wilson_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Interval kepercayaan Wilson untuk proporsi (lebih baik dari Wald untuk p kecil)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(0.5 + confidence / 2.0)
    p = k / n
    denom = 1.0 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (float(max(0.0, center - half)), float(min(1.0, center + half)))


def class_distribution(df: pd.DataFrame, by: str | None = None,
                       target: str = TARGET_COLUMN) -> pd.DataFrame:
    """Distribusi kelas (jumlah, proporsi fraud, CI Wilson 95%), opsional per ``by``
    (mis. 'tahun' atau 'sektor')."""
    y = pd.to_numeric(df[target], errors="coerce")
    if by is None:
        groups = [("Semua", y.dropna())]
    else:
        yy_all = pd.to_numeric(df[target], errors="coerce")
        groups = [(str(k), pd.to_numeric(sub[target], errors="coerce").dropna())
                  for k, sub in df.assign(**{target: yy_all}).groupby(by, sort=True)]
    rows = []
    for name, yy in groups:
        n = int(yy.size)
        k = int((yy == 1).sum())
        lo, hi = wilson_interval(k, n)
        rows.append({"kelompok": name, "n": n, "fraud": k, "non_fraud": n - k,
                     "proporsi_fraud": k / n if n else np.nan, "ci95_bawah": lo, "ci95_atas": hi})
    return pd.DataFrame(rows)


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """Koreksi FDR Benjamini-Hochberg. NaN dipertahankan sebagai NaN."""
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    m = int(valid.sum())
    if m == 0:
        return q
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order] * m / (np.arange(1, m + 1))
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]  # monoton non-menurun
    adj = np.empty(m)
    adj[order] = np.clip(ranked, 0.0, 1.0)
    q[valid] = adj
    return q


def compare_groups(df: pd.DataFrame, columns: Sequence[str] | None = None,
                   target: str = TARGET_COLUMN) -> pd.DataFrame:
    """Bandingkan kelompok fraud vs non-fraud dengan uji Mann-Whitney U.

    * ``efek_rank_biserial`` = 2U/(n1*n2) - 1 (sama dengan Cliff's delta), rentang -1..1;
      positif berarti nilai kelompok fraud cenderung lebih besar.
    * ``q_value`` = p-value terkoreksi Benjamini-Hochberg (kontrol FDR).
    """
    cols = list(columns) if columns is not None else numeric_columns(df, exclude=[target])
    y = pd.to_numeric(df[target], errors="coerce")
    rows = []
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce")
        a = x[(y == 1) & x.notna()].to_numpy(dtype=float)
        b = x[(y == 0) & x.notna()].to_numpy(dtype=float)
        row = {"variabel": c, "n_fraud": int(a.size), "n_non_fraud": int(b.size),
               "median_fraud": float(np.median(a)) if a.size else np.nan,
               "median_non_fraud": float(np.median(b)) if b.size else np.nan,
               "U": np.nan, "p_value": np.nan, "efek_rank_biserial": np.nan}
        if a.size >= 3 and b.size >= 3 and (np.ptp(np.concatenate([a, b])) > 0):
            res = stats.mannwhitneyu(a, b, alternative="two-sided", method="asymptotic")
            row["U"] = float(res.statistic)
            row["p_value"] = float(res.pvalue)
            row["efek_rank_biserial"] = float(2.0 * res.statistic / (a.size * b.size) - 1.0)
        rows.append(row)
    out = pd.DataFrame(rows)
    out["q_value"] = benjamini_hochberg(out["p_value"].to_numpy())
    return out.sort_values("p_value", na_position="last").reset_index(drop=True)
