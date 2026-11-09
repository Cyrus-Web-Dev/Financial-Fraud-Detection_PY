"""Analisis digit: Hukum Benford (digit pertama) dan uji digit terakhir.

Dasar teori
-----------
* Digit pertama (Benford): P(D1 = d) = log10(1 + 1/d), d = 1..9.
* Digit terakhir: pada angka yang cukup besar, digit terakhir mendekati
  seragam, P(D = d) = 1/10. Penyimpangan (mis. terlalu banyak 0 dan 5)
  mengindikasikan pembulatan atau angka karangan.

Catatan statistik penting (untuk dituliskan di skripsi)
-------------------------------------------------------
* Uji chi-square sangat sensitif pada sampel besar: dengan n ribuan,
  penyimpangan sekecil apa pun menghasilkan p-value mendekati 0. Karena itu
  Nigrini (2012) menyarankan *Mean Absolute Deviation* (MAD) sebagai ukuran
  kesesuaian utama; chi-square dan z-statistic dilaporkan sebagai pelengkap.
* Ambang MAD digit pertama (Nigrini, 2012):
  0 - 0.006 sangat sesuai | 0.006 - 0.012 cukup sesuai |
  0.012 - 0.015 marginal  | > 0.015 tidak sesuai.
* Hukum Benford berlaku untuk data yang mencakup beberapa orde besaran dan
  tidak dibatasi/ditetapkan (bukan nomor telepon, dsb.). Data laporan
  keuangan umumnya memenuhi syarat ini; tetapi ketidaksesuaian BUKAN bukti
  fraud, hanya indikasi untuk diperiksa lebih lanjut.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from src.data.schema import CURRENT_COLUMNS, TARGET_COLUMN

BENFORD_FIRST: np.ndarray = np.log10(1.0 + 1.0 / np.arange(1, 10))
UNIFORM_LAST: np.ndarray = np.full(10, 0.1)

MAD_THRESHOLDS = ((0.006, "Sangat sesuai"), (0.012, "Cukup sesuai"), (0.015, "Marginal"))
MIN_N_WARNING: int = 300  # di bawah ini hasil uji kurang andal


@dataclass
class DigitTestResult:
    kind: str                     # "first_digit" atau "last_digit"
    digits: list[int]
    n: int
    observed_count: list[int]
    observed_prop: list[float]
    expected_prop: list[float]
    z_scores: list[float]
    chi2: float
    dof: int
    p_value: float
    mad: float
    conformity: str
    small_sample: bool

    def to_dict(self) -> dict:
        return asdict(self)


def extract_first_digits(values: Iterable[float]) -> np.ndarray:
    """Digit pertama (1-9) dari nilai bukan nol. Nilai NaN/inf/0 diabaikan."""
    x = np.abs(np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float))
    x = x[np.isfinite(x) & (x > 0)]
    if x.size == 0:
        return np.array([], dtype=int)
    e = np.floor(np.log10(x))
    m = x / np.power(10.0, e)
    # koreksi galat floating-point di sekitar pangkat 10
    e = np.where(m >= 10.0, e + 1, np.where(m < 1.0, e - 1, e))
    m = x / np.power(10.0, e)
    d = np.floor(m).astype(int)
    return np.clip(d, 1, 9)


def extract_last_digits(values: Iterable[float], min_value: float = 100.0) -> np.ndarray:
    """Digit terakhir (0-9) dari nilai bulat dengan |x| >= ``min_value``.

    Nilai non-integer dibuang karena digit terakhir hanya bermakna pada
    angka yang dilaporkan sebagai bilangan bulat.
    """
    x = np.abs(np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float))
    x = x[np.isfinite(x) & (x >= min_value)]
    x = x[np.isclose(x, np.round(x), atol=1e-9)]
    return (np.round(x).astype(np.int64) % 10).astype(int)


def _conformity_label(mad: float) -> str:
    for upper, label in MAD_THRESHOLDS:
        if mad <= upper:
            return label
    return "Tidak sesuai"


def _digit_test(digits: np.ndarray, categories: Sequence[int], expected: np.ndarray, kind: str) -> DigitTestResult:
    n = int(digits.size)
    k = len(categories)
    counts = np.array([(digits == c).sum() for c in categories], dtype=int)
    if n == 0:
        nan = float("nan")
        return DigitTestResult(kind, list(categories), 0, counts.tolist(), [nan] * k,
                               expected.tolist(), [nan] * k, nan, k - 1, nan, nan, "Data tidak cukup", True)
    obs = counts / n
    exp_counts = expected * n
    chi2 = float(((counts - exp_counts) ** 2 / exp_counts).sum())
    dof = k - 1
    p_value = float(stats.chi2.sf(chi2, dof))
    mad = float(np.abs(obs - expected).mean())

    # z-statistic per digit dengan koreksi kontinuitas (Nigrini)
    num = np.maximum(np.abs(obs - expected) - 1.0 / (2.0 * n), 0.0)
    z = num / np.sqrt(expected * (1.0 - expected) / n)

    if kind == "first_digit":
        label = _conformity_label(mad)
    else:  # digit terakhir: tidak ada ambang MAD baku, gunakan p-value
        label = "Seragam (tidak ada bukti penyimpangan)" if p_value >= 0.05 else "Tidak seragam (p < 0,05)"

    return DigitTestResult(kind, list(categories), n, counts.tolist(), obs.tolist(), expected.tolist(),
                           z.tolist(), chi2, dof, p_value, mad, label, n < MIN_N_WARNING)


def benford_first_digit_test(values: Iterable[float]) -> DigitTestResult:
    """Uji kesesuaian digit pertama terhadap Hukum Benford."""
    return _digit_test(extract_first_digits(values), list(range(1, 10)), BENFORD_FIRST, "first_digit")


def last_digit_test(values: Iterable[float], min_value: float = 100.0) -> DigitTestResult:
    """Uji keseragaman digit terakhir (H0: setiap digit 0-9 berpeluang 10%)."""
    return _digit_test(extract_last_digits(values, min_value), list(range(10)), UNIFORM_LAST, "last_digit")


def _pool(df: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    parts = [pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float) for c in columns if c in df.columns]
    return np.concatenate(parts) if parts else np.array([], dtype=float)


def benford_by_column(df: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Ringkasan uji digit pertama per kolom (MAD, chi-square, p-value, label)."""
    cols = [c for c in (columns or CURRENT_COLUMNS) if c in df.columns]
    rows = []
    for c in cols:
        r = benford_first_digit_test(pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float))
        rows.append({"kolom": c, "n": r.n, "mad": r.mad, "chi2": r.chi2, "p_value": r.p_value,
                     "kesesuaian": r.conformity})
    return pd.DataFrame(rows)


def benford_by_group(df: pd.DataFrame, columns: Sequence[str] | None = None,
                     target: str = TARGET_COLUMN) -> dict[str, dict]:
    """Uji digit pertama dan terakhir pada gabungan kolom, dipisah per kelas label.

    Mengembalikan kamus ``{"semua": {...}, "non_fraud": {...}, "fraud": {...}}``
    dengan hasil ``first_digit`` dan ``last_digit`` (berupa dict).
    """
    cols = [c for c in (columns or CURRENT_COLUMNS) if c in df.columns]
    groups: dict[str, pd.DataFrame] = {"semua": df}
    if target in df.columns:
        y = pd.to_numeric(df[target], errors="coerce")
        groups["non_fraud"] = df[y == 0]
        groups["fraud"] = df[y == 1]
    out: dict[str, dict] = {}
    for name, sub in groups.items():
        pooled = _pool(sub, cols)
        out[name] = {
            "n_baris": int(len(sub)),
            "first_digit": benford_first_digit_test(pooled).to_dict(),
            "last_digit": last_digit_test(pooled).to_dict(),
        }
    return out


def rounding_features(df: pd.DataFrame, columns: Sequence[str] | None = None,
                      base: int = 1000) -> pd.DataFrame:
    """Fitur per baris: porsi pos yang dibulatkan ke kelipatan ``base``.

    Hanya pos dengan |nilai| >= base yang dihitung (agar nilai kecil tidak
    otomatis dianggap bulat). Hasil 0 jika tidak ada pos yang memenuhi syarat.
    Cocok sebagai fitur digit-preference pada tahap pemodelan.
    """
    cols = [c for c in (columns or CURRENT_COLUMNS) if c in df.columns]
    mat = np.column_stack([pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float) for c in cols])
    eligible = np.isfinite(mat) & (np.abs(mat) >= base) & np.isclose(mat, np.round(mat), atol=1e-9)
    safe = np.where(eligible, np.round(mat), 0.0)
    is_round = eligible & (np.mod(safe, base) == 0)
    n_elig = eligible.sum(axis=1)
    share = np.divide(is_round.sum(axis=1), n_elig, out=np.zeros(len(df)), where=n_elig > 0)
    return pd.DataFrame({f"round_share_{base}": share}, index=df.index)
