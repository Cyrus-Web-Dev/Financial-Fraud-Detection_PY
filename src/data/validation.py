"""Validasi dataset laporan keuangan.

``validate_dataset`` mengembalikan (DataFrame hasil koersi tipe, laporan).
Laporan berisi daftar temuan dengan tingkat keparahan:

* ``error``   : data tidak bisa dipakai sebelum diperbaiki
* ``warning`` : data bisa dipakai, tetapi perlu perhatian (mis. missing, outlier)
* ``info``    : informasi tambahan
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.data.quality import DEFAULT_OUTLIER_THRESHOLD, outlier_mask
from src.data.schema import (
    ALL_COLUMNS,
    IDENTIFIER_COLUMNS,
    NUMERIC_COLUMNS,
    REQUIRED_COLUMNS_FOR_PREDICTION,
    TARGET_COLUMN,
)

ERROR, WARNING, INFO = "error", "warning", "info"

# Batas yang dipakai (didokumentasikan agar bisa dijelaskan di skripsi)
MISSING_ERROR_SHARE = 0.40       # kolom dengan >40% missing dianggap tidak layak
BALANCE_TOLERANCE = 0.01         # toleransi persamaan neraca (1% dari total aset)
CONTINUITY_TOLERANCE = 0.01      # toleransi kesesuaian nilai t-1 dengan tahun lalu
MIN_EVENTS_WARNING = 30          # jumlah fraud minimum yang disarankan untuk modeling
YEAR_RANGE = (1990, 2100)

# Penyebut rasio: harus > 0 agar rasio keuangan terdefinisi
DENOMINATOR_COLUMNS = ["total_assets", "revenue", "total_assets_prev", "revenue_prev"]
# Pos yang secara akuntansi tidak boleh negatif
NON_NEGATIVE_COLUMNS = [
    "cash", "receivables", "inventory", "ppe_net", "current_liabilities", "long_term_debt",
    "total_liabilities", "cogs", "sga", "depreciation", "interest_expense", "market_value_equity",
    "current_assets", "receivables_prev", "ppe_net_prev", "current_liabilities_prev",
    "long_term_debt_prev", "cogs_prev", "sga_prev", "depreciation_prev", "current_assets_prev",
]


@dataclass
class Issue:
    severity: str
    code: str
    message: str
    column: str | None = None
    count: int | None = None
    share: float | None = None


@dataclass
class ValidationReport:
    issues: list[Issue] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def add(self, severity: str, code: str, message: str, column: str | None = None,
            count: int | None = None, share: float | None = None) -> None:
        self.issues.append(Issue(severity, code, message, column, count, share))

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == ERROR]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == WARNING]

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "issues": [asdict(i) for i in self.issues],
            "summary": self.summary,
        }


def _pct(count: int, total: int) -> float:
    return float(count) / total if total else 0.0


def _check_columns(df: pd.DataFrame, report: ValidationReport, require_target: bool) -> bool:
    """Kembalikan True jika semua kolom wajib tersedia."""
    required = list(REQUIRED_COLUMNS_FOR_PREDICTION)
    if require_target:
        required.append(TARGET_COLUMN)
    missing = [c for c in required if c not in df.columns]
    if missing:
        report.add(ERROR, "KOLOM_HILANG",
                   f"Kolom wajib tidak ditemukan: {', '.join(missing)}", count=len(missing))
    extra = [c for c in df.columns if c not in ALL_COLUMNS]
    if extra:
        report.add(INFO, "KOLOM_TAMBAHAN",
                   f"{len(extra)} kolom tambahan akan diabaikan: {', '.join(extra[:8])}"
                   + (" ..." if len(extra) > 8 else ""), count=len(extra))
    return not missing


def _coerce_numeric(df: pd.DataFrame, report: ValidationReport) -> pd.DataFrame:
    out = df.copy()
    for col in NUMERIC_COLUMNS:
        if col not in out.columns or pd.api.types.is_numeric_dtype(out[col]):
            continue
        original_notna = out[col].notna()
        coerced = pd.to_numeric(out[col], errors="coerce")
        bad = int((original_notna & coerced.isna()).sum())
        if bad:
            sample = out.loc[original_notna & coerced.isna(), col].astype(str).head(3).tolist()
            report.add(ERROR, "BUKAN_ANGKA",
                       f"Kolom '{col}' berisi {bad} nilai non-numerik (contoh: {', '.join(sample)}).",
                       column=col, count=bad, share=_pct(bad, len(out)))
        out[col] = coerced
    return out


def _check_year(df: pd.DataFrame, report: ValidationReport) -> pd.DataFrame:
    out = df.copy()
    year = pd.to_numeric(out["tahun"], errors="coerce")
    n_bad = int(year.isna().sum())
    if n_bad:
        report.add(ERROR, "TAHUN_TIDAK_VALID", f"Kolom 'tahun' memiliki {n_bad} nilai kosong atau non-numerik.",
                   column="tahun", count=n_bad, share=_pct(n_bad, len(out)))
        return out
    if not np.all(np.mod(year.to_numpy(dtype=float), 1) == 0):
        report.add(ERROR, "TAHUN_TIDAK_VALID", "Kolom 'tahun' harus berupa bilangan bulat.", column="tahun")
        return out
    lo, hi = YEAR_RANGE
    out_of_range = int(((year < lo) | (year > hi)).sum())
    if out_of_range:
        report.add(ERROR, "TAHUN_DI_LUAR_RENTANG", f"{out_of_range} baris memiliki tahun di luar {lo}-{hi}.",
                   column="tahun", count=out_of_range)
    out["tahun"] = year.astype(int)
    return out


def _check_missing(df: pd.DataFrame, report: ValidationReport) -> None:
    cols = [c for c in NUMERIC_COLUMNS + IDENTIFIER_COLUMNS if c in df.columns]
    miss = df[cols].isna().sum()
    n = len(df)
    for col, cnt in miss.items():
        if cnt == 0:
            continue
        share = _pct(int(cnt), n)
        sev = ERROR if share > MISSING_ERROR_SHARE else WARNING
        report.add(sev, "MISSING", f"Kolom '{col}' memiliki {int(cnt)} nilai kosong ({share:.1%}).",
                   column=col, count=int(cnt), share=share)
    rows_with_missing = int(df[cols].isna().any(axis=1).sum())
    report.summary["baris_dengan_missing"] = rows_with_missing
    report.summary["persen_baris_dengan_missing"] = _pct(rows_with_missing, n)


def _check_duplicates(df: pd.DataFrame, report: ValidationReport) -> None:
    if df[["company_id", "tahun"]].isna().any().any():
        return
    dup = int(df.duplicated(subset=["company_id", "tahun"]).sum())
    if dup:
        report.add(ERROR, "DUPLIKAT",
                   f"{dup} baris duplikat pada kombinasi (company_id, tahun). Setiap perusahaan hanya boleh "
                   "punya satu baris per tahun.", count=dup, share=_pct(dup, len(df)))
    full_dup = int(df.duplicated().sum())
    if full_dup and not dup:
        report.add(WARNING, "BARIS_IDENTIK", f"{full_dup} baris identik.", count=full_dup)


def _check_ranges(df: pd.DataFrame, report: ValidationReport) -> None:
    n = len(df)
    for col in DENOMINATOR_COLUMNS:
        if col in df.columns:
            bad = int((df[col] <= 0).sum())
            if bad:
                report.add(ERROR, "PENYEBUT_NON_POSITIF",
                           f"Kolom '{col}' memiliki {bad} nilai <= 0; rasio keuangan tidak dapat dihitung.",
                           column=col, count=bad, share=_pct(bad, n))
    for col in NON_NEGATIVE_COLUMNS:
        if col in df.columns:
            bad = int((df[col] < 0).sum())
            if bad:
                report.add(WARNING, "NILAI_NEGATIF",
                           f"Kolom '{col}' memiliki {bad} nilai negatif yang tidak lazim.",
                           column=col, count=bad, share=_pct(bad, n))


def _check_accounting_identities(df: pd.DataFrame, report: ValidationReport) -> None:
    n = len(df)
    ta = df["total_assets"].where(df["total_assets"] > 0)

    gap = (df["total_assets"] - df["total_liabilities"] - df["equity"]).abs() / ta
    bad = int((gap > BALANCE_TOLERANCE).sum())
    if bad:
        report.add(WARNING, "NERACA_TIDAK_SEIMBANG",
                   f"{bad} baris: Total Aset != Total Liabilitas + Ekuitas (selisih > {BALANCE_TOLERANCE:.0%}).",
                   count=bad, share=_pct(bad, n))

    gap = (df["total_liabilities"] - df["current_liabilities"] - df["long_term_debt"]).abs() / ta
    bad = int((gap > BALANCE_TOLERANCE).sum())
    if bad:
        report.add(WARNING, "LIABILITAS_TIDAK_SESUAI",
                   f"{bad} baris: Total Liabilitas != Jangka Pendek + Jangka Panjang.",
                   count=bad, share=_pct(bad, n))

    parts = df["cash"] + df["receivables"] + df["inventory"]
    bad = int((parts > df["current_assets"] * (1 + BALANCE_TOLERANCE)).sum())
    if bad:
        report.add(WARNING, "ASET_LANCAR_TIDAK_SESUAI",
                   f"{bad} baris: Kas + Piutang + Persediaan melebihi Aset Lancar.",
                   count=bad, share=_pct(bad, n))


def _check_panel_continuity(df: pd.DataFrame, report: ValidationReport) -> None:
    """Nilai *_prev pada tahun t harus sama dengan nilai tahun t-1 perusahaan yang sama."""
    if df[["company_id", "tahun"]].isna().any().any() or df.duplicated(["company_id", "tahun"]).any():
        return
    for col in ("total_assets", "revenue"):
        last_year = df[["company_id", "tahun", col]].copy()
        last_year["tahun"] = last_year["tahun"] + 1
        last_year = last_year.rename(columns={col: "_last"})
        merged = df[["company_id", "tahun", f"{col}_prev"]].merge(
            last_year, on=["company_id", "tahun"], how="inner")
        comparable = merged.dropna(subset=[f"{col}_prev", "_last"])
        if comparable.empty:
            continue
        denom = comparable["_last"].abs().where(comparable["_last"].abs() > 0)
        rel = (comparable[f"{col}_prev"] - comparable["_last"]).abs() / denom
        bad = int((rel > CONTINUITY_TOLERANCE).sum())
        if bad:
            report.add(WARNING, "KONTINUITAS_PANEL",
                       f"{bad} dari {len(comparable)} pasangan tahun: '{col}_prev' tidak sama dengan nilai "
                       f"'{col}' tahun sebelumnya (selisih > {CONTINUITY_TOLERANCE:.0%}).",
                       column=f"{col}_prev", count=bad, share=_pct(bad, len(comparable)))


def _check_outliers(df: pd.DataFrame, report: ValidationReport, threshold: float) -> None:
    mask = outlier_mask(df, threshold)
    counts = mask.sum(axis=0)
    n = len(df)
    total = int(counts.sum())
    report.summary["jumlah_sel_outlier"] = total
    report.summary["baris_dengan_outlier"] = int(mask.any(axis=1).sum())
    for col, cnt in counts[counts > 0].sort_values(ascending=False).items():
        report.add(WARNING, "OUTLIER",
                   f"Kolom '{col}' memiliki {int(cnt)} sel yang sangat menyimpang (modified z-score > {threshold:g}).",
                   column=col, count=int(cnt), share=_pct(int(cnt), n))


def _check_target(df: pd.DataFrame, report: ValidationReport) -> None:
    y = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    n_nan = int(y.isna().sum())
    if n_nan:
        report.add(ERROR, "LABEL_KOSONG", f"Kolom '{TARGET_COLUMN}' memiliki {n_nan} nilai kosong/non-numerik.",
                   column=TARGET_COLUMN, count=n_nan)
        return
    if not set(y.unique()) <= {0, 1}:
        report.add(ERROR, "LABEL_TIDAK_BINER", f"Kolom '{TARGET_COLUMN}' hanya boleh berisi 0 dan 1.",
                   column=TARGET_COLUMN)
        return
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    report.summary["jumlah_fraud"] = n_pos
    report.summary["jumlah_non_fraud"] = n_neg
    report.summary["proporsi_fraud"] = _pct(n_pos, len(y))
    if n_pos == 0 or n_neg == 0:
        report.add(ERROR, "LABEL_SATU_KELAS", "Label hanya berisi satu kelas; model tidak dapat dilatih.")
    elif n_pos < MIN_EVENTS_WARNING:
        report.add(WARNING, "FRAUD_SEDIKIT",
                   f"Hanya {n_pos} observasi fraud (< {MIN_EVENTS_WARNING}); estimasi model akan sangat tidak stabil.",
                   count=n_pos)
    else:
        report.add(INFO, "IMBALANCE",
                   f"Proporsi fraud {_pct(n_pos, len(y)):.2%} ({n_pos} dari {len(y)}). "
                   "Perlu penanganan ketidakseimbangan kelas dan metrik yang sesuai.", count=n_pos)


def validate_dataset(
    df: pd.DataFrame,
    require_target: bool = False,
    outlier_threshold: float = DEFAULT_OUTLIER_THRESHOLD,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Validasi dataset dan kembalikan (data dengan tipe sudah dikoersi, laporan).

    Parameters
    ----------
    require_target : True untuk data latih/evaluasi (kolom ``is_fraud`` wajib).
    """
    report = ValidationReport()
    n_rows, n_cols = df.shape
    report.summary.update({"n_baris": int(n_rows), "n_kolom": int(n_cols)})

    if n_rows == 0:
        report.add(ERROR, "DATA_KOSONG", "Dataset tidak memiliki baris.")
        return df.copy(), report

    if not _check_columns(df, report, require_target):
        return df.copy(), report  # tidak bisa lanjut tanpa kolom wajib

    clean = _coerce_numeric(df, report)
    if TARGET_COLUMN in clean.columns:
        clean[TARGET_COLUMN] = pd.to_numeric(clean[TARGET_COLUMN], errors="coerce")
    clean = _check_year(clean, report)

    if report.errors:  # kesalahan tipe: pemeriksaan lanjutan tidak bermakna
        return clean, report

    report.summary["n_perusahaan"] = int(clean["company_id"].nunique())
    report.summary["rentang_tahun"] = [int(clean["tahun"].min()), int(clean["tahun"].max())]

    _check_missing(clean, report)
    _check_duplicates(clean, report)
    _check_ranges(clean, report)
    if not any(i.code == "PENYEBUT_NON_POSITIF" for i in report.issues):
        _check_accounting_identities(clean, report)
    _check_panel_continuity(clean, report)
    _check_outliers(clean, report, outlier_threshold)
    if TARGET_COLUMN in clean.columns and (require_target or clean[TARGET_COLUMN].notna().any()):
        _check_target(clean, report)

    if clean[NUMERIC_COLUMNS].isna().any().any():
        report.add(INFO, "IMPUTASI",
                   "Missing value akan ditangani pada tahap feature engineering (imputasi hanya dari data latih "
                   "agar tidak terjadi data leakage).")
    return clean, report
