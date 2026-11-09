"""Generator data panel laporan keuangan sintetis dengan label fraud.

Tujuan: menyediakan data demo yang realistis sampai data asli (mis. IDX)
tersedia. Pipeline berikutnya hanya bergantung pada skema di ``schema.py``,
sehingga data asli tinggal menggantikan file CSV ini.

Prinsip desain
--------------
* Laporan keuangan disimulasikan per perusahaan dari tahun ke tahun
  (t-1 -> t), sehingga kolom ``*_prev`` konsisten dengan tahun sebelumnya.
* Persamaan neraca selalu terpenuhi: Total Aset = Aset Lancar + Aset Tetap
  + Aset Tidak Lancar Lain, dan Ekuitas = Total Aset - Total Liabilitas.
* Fraud TIDAK ditanam sebagai rasio tertentu. Fraud dimodelkan sebagai
  manipulasi akrual (pendapatan dan piutang digelembungkan, penyusutan
  diperlambat, leverage naik) dengan intensitas acak, sehingga rasio
  seperti DSRI, TATA, dan LVGI terdampak secara alami dan masih ada
  tumpang tindih dengan perusahaan non-fraud.
* Angka fraud sebagian dibulatkan ke ribuan (digit preference), sehingga
  uji digit terakhir dan uji Benford punya sinyal untuk dideteksi.
* Data memiliki struktur panel; saat validasi, bagi data berdasarkan
  perusahaan atau waktu agar tidak terjadi kebocoran (data leakage).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from src.config import (
    RANDOM_SEED,
    SYNTH_FRAUD_PREVALENCE,
    SYNTH_N_FIRMS,
    SYNTH_YEARS,
)
from src.data.schema import ALL_COLUMNS, NUMERIC_COLUMNS, TARGET_COLUMN

# Parameter per sektor: (rata2 gross margin, rata2 leverage, rata2 porsi aset tetap,
# rata2 pertumbuhan pendapatan tahunan)
SECTOR_PARAMS: dict[str, dict[str, float]] = {
    "Manufaktur": {"gm": 0.28, "lev": 0.45, "ppe": 0.30, "growth": 0.06},
    "Perdagangan": {"gm": 0.22, "lev": 0.50, "ppe": 0.15, "growth": 0.07},
    "Properti": {"gm": 0.40, "lev": 0.50, "ppe": 0.25, "growth": 0.05},
    "Pertambangan": {"gm": 0.35, "lev": 0.45, "ppe": 0.36, "growth": 0.05},
    "Jasa": {"gm": 0.45, "lev": 0.40, "ppe": 0.20, "growth": 0.08},
    "Infrastruktur": {"gm": 0.30, "lev": 0.60, "ppe": 0.38, "growth": 0.04},
}

# Proporsi perusahaan non-fraud yang "agresif tapi sah" per tahun
AGGRESSIVE_SHARE: float = 0.12

# Pos yang mengalami "digit preference" (pembulatan ke ribuan) pada baris fraud.
# Hanya pos laba rugi / arus kas agar persamaan neraca tetap exact.
ROUNDED_IF_FRAUD: tuple[str, ...] = ("revenue", "net_income", "operating_cash_flow")


def _round_int(x: np.ndarray) -> np.ndarray:
    """Bulatkan ke bilangan bulat (satuan Rupiah juta), tetap bertipe float."""
    return np.round(x)


def _initial_state(n: int, sector_idx: np.ndarray, sectors: Sequence[str],
                   rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Bangun keadaan awal perusahaan (tahun dasar, sebelum tahun pertama data)."""
    gm_mean = np.array([SECTOR_PARAMS[s]["gm"] for s in sectors])[sector_idx]
    lev_mean = np.array([SECTOR_PARAMS[s]["lev"] for s in sectors])[sector_idx]
    ppe_mean = np.array([SECTOR_PARAMS[s]["ppe"] for s in sectors])[sector_idx]

    ta = _round_int(rng.lognormal(mean=np.log(1.5e6), sigma=1.4, size=n))
    ta = np.maximum(ta, 5e4)

    cash = _round_int(ta * rng.uniform(0.03, 0.12, n))
    rec = _round_int(ta * rng.uniform(0.05, 0.18, n))
    inv = _round_int(ta * rng.uniform(0.03, 0.15, n))
    oca = _round_int(ta * rng.uniform(0.01, 0.04, n))
    ppe = _round_int(ta * np.clip(rng.normal(ppe_mean, 0.06), 0.10, 0.40))
    ca = cash + rec + inv + oca
    # Aset tidak lancar lain = sisa (selalu positif karena total porsi <= 0.89)
    onc = ta - ca - ppe
    onc = np.maximum(onc, 0.0)
    ta = ca + ppe + onc  # pastikan persamaan neraca exact

    dso = rng.uniform(35, 90, n)
    rev = _round_int(rec * 365.0 / dso)
    gm = np.clip(rng.normal(gm_mean, 0.05), 0.08, 0.70)
    cogs = _round_int(rev * (1.0 - gm))
    sga = _round_int(rev * rng.uniform(0.05, 0.15, n))
    dep = _round_int(ppe * rng.uniform(0.04, 0.10, n))

    lev = np.clip(rng.normal(lev_mean, 0.10), 0.15, 0.80)
    tl = _round_int(ta * lev)
    cl = _round_int(tl * rng.uniform(0.45, 0.85, n))
    ltd = tl - cl
    eq = ta - tl
    re = _round_int(eq * rng.uniform(0.3, 0.8, n))

    return {
        "ta": ta, "ca": ca, "cash": cash, "rec": rec, "inv": inv, "oca": oca,
        "ppe": ppe, "onc": onc, "rev": rev, "cogs": cogs, "sga": sga,
        "dep": dep, "tl": tl, "cl": cl, "ltd": ltd, "eq": eq, "re": re,
    }


def _step(prev: dict[str, np.ndarray], fraud: np.ndarray, s: np.ndarray,
          growth: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Simulasikan satu tahun (t-1 -> t).

    ``fraud`` bernilai 0/1 dan ``s`` adalah intensitas manipulasi (0.1 - 1.0).
    Efek manipulasi = fraud * s * (besaran acak), jadi baris non-fraud
    tidak terpengaruh sama sekali.
    """
    n = len(fraud)
    f = fraud.astype(float) * s  # intensitas manipulasi; 0 untuk non-fraud
    # Perusahaan non-fraud yang "agresif tapi sah": pertumbuhan, piutang, dan
    # utang naik cepat tanpa manipulasi. Ini membuat kelas saling tumpang
    # tindih, seperti pada data nyata (false positive yang wajar).
    aggressive = ((fraud == 0) & (rng.random(n) < AGGRESSIVE_SHARE)).astype(float)
    a = aggressive * rng.uniform(0.3, 0.9, n)
    fa = f + a  # dipakai untuk pos yang juga naik pada perusahaan agresif

    # --- Laba rugi ---------------------------------------------------------
    g = rng.normal(growth, 0.08) + fa * rng.uniform(0.04, 0.14, n)
    rev = np.maximum(_round_int(prev["rev"] * (1.0 + g)), 1.0)

    gm_prev = 1.0 - prev["cogs"] / np.maximum(prev["rev"], 1.0)
    gm = np.clip(gm_prev + rng.normal(0.0, 0.02, n) - f * rng.uniform(0.0, 0.03, n),
                 0.03, 0.80)
    cogs = _round_int(rev * (1.0 - gm))

    sga_growth = 0.8 * g + rng.normal(0.0, 0.04, n) + fa * 0.04
    sga = np.maximum(_round_int(prev["sga"] * (1.0 + sga_growth)), 1.0)

    # --- Aset --------------------------------------------------------------
    rec = _round_int(prev["rec"] * (1.0 + g) *
                     (1.0 + rng.normal(0.0, 0.07, n) + fa * rng.uniform(0.10, 0.30, n)))
    inv = _round_int(prev["inv"] * (1.0 + g) *
                     (1.0 + rng.normal(0.0, 0.09, n) + fa * 0.08))
    ppe = _round_int(prev["ppe"] * (1.0 + rng.normal(0.04, 0.08, n) + fa * 0.08))
    cash = _round_int(prev["cash"] * (1.0 + rng.normal(0.03, 0.15, n)))
    oca = _round_int(prev["oca"] * (1.0 + rng.normal(0.03, 0.10, n)))
    rec, inv, ppe, cash, oca = (np.maximum(v, 1.0) for v in (rec, inv, ppe, cash, oca))
    ca = cash + rec + inv + oca
    onc = np.maximum(_round_int(prev["onc"] * (1.0 + rng.normal(0.03, 0.05, n))), 0.0)
    ta = ca + ppe + onc

    # Penyusutan: pada fraud diperlambat (umur ekonomis diperpanjang)
    dep = np.maximum(_round_int(ppe * rng.uniform(0.04, 0.10, n) * (1.0 - f * 0.25)), 1.0)

    # --- Liabilitas dan ekuitas -------------------------------------------
    tl = prev["tl"] * (1.0 + rng.normal(0.04, 0.09, n) + fa * rng.uniform(0.04, 0.14, n))
    tl = _round_int(np.clip(tl, 0.10 * ta, 1.15 * ta))
    share_prev = prev["cl"] / np.maximum(prev["tl"], 1.0)
    share = np.clip(share_prev + rng.normal(0.0, 0.03, n), 0.30, 0.90)
    cl = _round_int(tl * share)
    ltd = tl - cl
    eq = ta - tl

    # --- Laba -------------------------------------------------------------
    ebit = rev - cogs - sga - dep
    interest = _round_int(ltd * rng.uniform(0.06, 0.11, n))
    pbt = ebit - interest
    ni = _round_int(np.where(pbt > 0, pbt * (1.0 - 0.22), pbt))

    # Arus kas operasi (metode tidak langsung, disederhanakan). Kenaikan piutang
    # dan persediaan mengurangi kas, sehingga fraud (piutang naik cepat) otomatis
    # menunjukkan kualitas laba yang rendah.
    cfo = _round_int(ni + dep - (rec - prev["rec"]) - (inv - prev["inv"])
                     + rng.normal(0.0, 0.01, n) * ta)
    # Laba ditahan tidak boleh melebihi ekuitas (komponen ekuitas lain tidak dimodelkan)
    re_raw = prev["re"] + ni
    re = _round_int(np.where(eq > 0, np.minimum(re_raw, 0.95 * eq), re_raw))
    mve = _round_int(np.maximum(eq, 0.05 * ta) * np.exp(rng.normal(0.2, 0.5, n))
                     * (1.0 + f * 0.3))

    # --- Digit preference pada baris fraud --------------------------------
    income_items = {"revenue": rev, "net_income": ni, "operating_cash_flow": cfo}
    do_round = (fraud == 1) & (rng.random(n) < 0.6 * s)
    for name in ROUNDED_IF_FRAUD:
        income_items[name] = np.where(
            do_round, np.round(income_items[name] / 1000.0) * 1000.0, income_items[name]
        )
    rev, ni, cfo = (income_items[k] for k in ROUNDED_IF_FRAUD)

    return {
        "ta": ta, "ca": ca, "cash": cash, "rec": rec, "inv": inv, "oca": oca,
        "ppe": ppe, "onc": onc, "rev": rev, "cogs": cogs, "sga": sga,
        "dep": dep, "tl": tl, "cl": cl, "ltd": ltd, "eq": eq, "re": re,
        "ebit": ebit, "interest": interest, "ni": ni, "cfo": cfo, "mve": mve,
    }


def generate_financial_panel(
    n_firms: int = SYNTH_N_FIRMS,
    years: Sequence[int] = SYNTH_YEARS,
    fraud_prevalence: float = SYNTH_FRAUD_PREVALENCE,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Hasilkan data panel (perusahaan x tahun) dengan skema ``ALL_COLUMNS``.

    Parameters
    ----------
    n_firms : jumlah perusahaan.
    years : tahun-tahun yang dihasilkan (urut naik, berurutan).
    fraud_prevalence : target proporsi baris fraud (0 < p < 0.5).
    seed : random seed agar hasil reproducible.
    """
    if n_firms < 10:
        raise ValueError("n_firms minimal 10.")
    if not 0.0 < fraud_prevalence < 0.5:
        raise ValueError("fraud_prevalence harus di antara 0 dan 0.5.")
    years = list(years)
    if len(years) == 0 or any(b - a != 1 for a, b in zip(years, years[1:])):
        raise ValueError("years harus berurutan, mis. (2018, 2019, 2020).")

    rng = np.random.default_rng(seed)
    sectors = list(SECTOR_PARAMS)
    sector_idx = rng.integers(0, len(sectors), n_firms)
    growth = np.array([SECTOR_PARAMS[s]["growth"] for s in sectors])[sector_idx]
    firm_ids = np.array([f"PRS-{i + 1:04d}" for i in range(n_firms)])
    sector_names = np.array(sectors)[sector_idx]

    # Kecenderungan fraud per perusahaan (heterogen), dinormalkan ke prevalensi target
    propensity = rng.lognormal(mean=0.0, sigma=0.5, size=n_firms)
    p_fraud = np.clip(fraud_prevalence * propensity / propensity.mean(), 0.0, 0.40)

    state = _initial_state(n_firms, sector_idx, sectors, rng)
    frames: list[pd.DataFrame] = []

    for year in years:
        fraud = (rng.random(n_firms) < p_fraud).astype(int)
        s = rng.beta(1.6, 1.6, n_firms) * 0.9 + 0.1
        prev = state
        cur = _step(prev, fraud, s, growth, rng)

        frame = pd.DataFrame({
            "company_id": firm_ids,
            "tahun": year,
            "sektor": sector_names,
            "total_assets": cur["ta"],
            "current_assets": cur["ca"],
            "cash": cur["cash"],
            "receivables": cur["rec"],
            "inventory": cur["inv"],
            "ppe_net": cur["ppe"],
            "current_liabilities": cur["cl"],
            "long_term_debt": cur["ltd"],
            "total_liabilities": cur["tl"],
            "equity": cur["eq"],
            "retained_earnings": cur["re"],
            "revenue": cur["rev"],
            "cogs": cur["cogs"],
            "sga": cur["sga"],
            "depreciation": cur["dep"],
            "ebit": cur["ebit"],
            "interest_expense": cur["interest"],
            "net_income": cur["ni"],
            "operating_cash_flow": cur["cfo"],
            "market_value_equity": cur["mve"],
            "total_assets_prev": prev["ta"],
            "current_assets_prev": prev["ca"],
            "receivables_prev": prev["rec"],
            "ppe_net_prev": prev["ppe"],
            "current_liabilities_prev": prev["cl"],
            "long_term_debt_prev": prev["ltd"],
            "revenue_prev": prev["rev"],
            "cogs_prev": prev["cogs"],
            "sga_prev": prev["sga"],
            "depreciation_prev": prev["dep"],
            TARGET_COLUMN: fraud,
        })
        frames.append(frame)
        # Keadaan t menjadi keadaan t-1 untuk tahun berikutnya
        state = {k: cur[k] for k in prev}

    df = pd.concat(frames, ignore_index=True)
    df = df[ALL_COLUMNS]
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    return df


def inject_data_issues(
    df: pd.DataFrame,
    missing_rate: float = 0.01,
    outlier_rate: float = 0.003,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Sisipkan missing value dan outlier ekstrem untuk demo halaman validasi.

    Hanya kolom numerik yang diubah; kolom identitas dan label tidak disentuh.
    """
    rng = np.random.default_rng(seed + 1)
    out = df.copy()
    values = np.array(out[NUMERIC_COLUMNS], dtype=float, copy=True)

    miss_mask = rng.random(values.shape) < missing_rate
    outlier_mask = (rng.random(values.shape) < outlier_rate) & ~miss_mask
    values[outlier_mask] *= rng.uniform(10.0, 30.0, size=int(outlier_mask.sum()))
    values[miss_mask] = np.nan

    out[NUMERIC_COLUMNS] = values
    return out
