"""Feature engineering: rasio keuangan, indeks Beneish, dan skor benchmark.

Dua keluaran utama
------------------
* ``compute_ratios(df)``           -> fitur untuk model (rasio + 8 indeks Beneish + fitur digit).
* ``compute_benchmark_scores(df)`` -> M-Score (Beneish) dan Z-Score (Altman) sebagai
  pembanding berbasis teori, BUKAN fitur model (skor komposit adalah kombinasi
  linear dari komponennya sehingga akan menimbulkan multikolinearitas sempurna).

Catatan metodologi untuk skripsi
--------------------------------
* Semua pembagian memakai ``safe_divide``: penyebut ~0 menghasilkan NaN (bukan inf).
* ROE, D/E, dan market-to-book hanya terdefinisi bila ekuitas > 0; selain itu NaN
  (rasio dengan ekuitas negatif tidak dapat ditafsirkan secara ekonomi).
* Koefisien Beneish (1999) diestimasi dari perusahaan manufaktur AS 1982-1992.
  Penerapan pada data Indonesia hanya sebagai pembanding; sebaiknya dilaporkan
  bersama model yang diestimasi ulang pada data sendiri.
* TATA di sini memakai definisi arus kas (Laba Bersih - Arus Kas Operasi) / Total Aset,
  versi modern yang lazim dipakai; definisi asli Beneish memakai perubahan akun
  modal kerja.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.data.schema import NUMERIC_COLUMNS
from src.evaluation.benford import rounding_features

# Koefisien model 8-variabel Beneish (1999)
BENEISH_COEF: dict[str, float] = {
    "intercept": -4.84, "dsri": 0.920, "gmi": 0.528, "aqi": 0.404, "sgi": 0.892,
    "depi": 0.115, "sgai": -0.172, "tata": 4.679, "lvgi": -0.327,
}
BENEISH_THRESHOLD: float = -1.78  # M > -1.78 -> indikasi manipulator (Beneish, 1999)

# Koefisien Altman Z-Score
ALTMAN_Z_COEF = {"x1": 1.2, "x2": 1.4, "x3": 3.3, "x4": 0.6, "x5": 1.0}          # perusahaan publik manufaktur
ALTMAN_Z2_COEF = {"intercept": 3.25, "x1": 6.56, "x2": 3.26, "x3": 6.72, "x4": 1.05}  # Z'' (emerging market/non-manufaktur)


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    label: str          # nama untuk tampilan (Bahasa Indonesia)
    group: str
    formula: str
    rationale: str
    expected_sign: int  # arah hubungan yang diharapkan dengan fraud: +1, -1, atau 0 (tidak pasti)


FEATURE_CATALOG: list[FeatureSpec] = [
    # --- Indeks Beneish ----------------------------------------------------
    FeatureSpec("dsri", "Indeks Piutang terhadap Penjualan (DSRI)", "Beneish",
                "(Piutang/Pendapatan)t / (Piutang/Pendapatan)t-1",
                "Piutang naik lebih cepat dari pendapatan dapat menandakan pendapatan fiktif.", +1),
    FeatureSpec("gmi", "Indeks Margin Kotor (GMI)", "Beneish",
                "Margin Kotor t-1 / Margin Kotor t",
                "Margin yang memburuk menambah tekanan untuk memanipulasi laba.", +1),
    FeatureSpec("aqi", "Indeks Kualitas Aset (AQI)", "Beneish",
                "[1-(AsetLancar+AsetTetap)/TotalAset]t / [..]t-1",
                "Kenaikan aset tak berwujud/lain-lain dapat menandakan penundaan biaya (kapitalisasi).", +1),
    FeatureSpec("sgi", "Indeks Pertumbuhan Penjualan (SGI)", "Beneish",
                "Pendapatan t / Pendapatan t-1",
                "Perusahaan bertumbuh pesat menghadapi tekanan untuk mempertahankan kinerja.", +1),
    FeatureSpec("depi", "Indeks Penyusutan (DEPI)", "Beneish",
                "Laju Penyusutan t-1 / Laju Penyusutan t",
                "Laju penyusutan yang melambat dapat menandakan perpanjangan umur ekonomis untuk menaikkan laba.", +1),
    FeatureSpec("sgai", "Indeks Beban SGA (SGAI)", "Beneish",
                "(SGA/Pendapatan)t / (SGA/Pendapatan)t-1",
                "Beban SGA yang naik tidak proporsional dapat menjadi sinyal prospek memburuk (hipotesis Beneish: positif; "
                "koefisien estimasi pada sampel AS negatif dan tidak signifikan).", +1),
    FeatureSpec("lvgi", "Indeks Leverage (LVGI)", "Beneish",
                "[(LiabLancar+LiabJangkaPanjang)/TotalAset]t / [..]t-1",
                "Kenaikan leverage meningkatkan insentif memenuhi perjanjian utang (hipotesis Beneish: positif; "
                "koefisien estimasi pada sampel AS negatif dan tidak signifikan).", +1),
    FeatureSpec("tata", "Akrual terhadap Total Aset (TATA)", "Beneish",
                "(Laba Bersih - Arus Kas Operasi) / Total Aset",
                "Laba yang tidak didukung kas (akrual tinggi) adalah sinyal kuat manajemen laba.", +1),
    # --- Likuiditas --------------------------------------------------------
    FeatureSpec("current_ratio", "Rasio Lancar", "Likuiditas", "Aset Lancar / Liabilitas Jangka Pendek",
                "Ukuran kemampuan memenuhi kewajiban jangka pendek.", 0),
    FeatureSpec("quick_ratio", "Rasio Cepat", "Likuiditas", "(Aset Lancar - Persediaan) / Liabilitas Jangka Pendek",
                "Likuiditas tanpa persediaan yang sulit diverifikasi.", 0),
    FeatureSpec("cash_ratio", "Rasio Kas", "Likuiditas", "Kas / Liabilitas Jangka Pendek",
                "Kas relatif sulit dimanipulasi dibanding akun akrual.", -1),
    # --- Solvabilitas ------------------------------------------------------
    FeatureSpec("debt_to_assets", "Liabilitas terhadap Aset", "Solvabilitas", "Total Liabilitas / Total Aset",
                "Leverage tinggi meningkatkan tekanan pemenuhan perjanjian utang.", +1),
    FeatureSpec("debt_to_equity", "Liabilitas terhadap Ekuitas (DER)", "Solvabilitas",
                "Total Liabilitas / Ekuitas (hanya jika Ekuitas > 0)",
                "Struktur modal agresif. Terdefinisi hanya untuk ekuitas positif.", +1),
    FeatureSpec("long_term_debt_to_assets", "Utang Jangka Panjang terhadap Aset", "Solvabilitas",
                "Liabilitas Jangka Panjang / Total Aset", "Beban utang struktural.", +1),
    FeatureSpec("interest_coverage", "Cakupan Bunga", "Solvabilitas", "EBIT / Beban Bunga",
                "Kemampuan membayar bunga; nilai rendah berarti tekanan keuangan.", -1),
    # --- Profitabilitas ----------------------------------------------------
    FeatureSpec("roa", "Return on Assets (ROA)", "Profitabilitas", "Laba Bersih / Total Aset",
                "Kinerja yang buruk maupun terlalu baik dibanding peers sama-sama mencurigakan.", 0),
    FeatureSpec("roe", "Return on Equity (ROE)", "Profitabilitas", "Laba Bersih / Ekuitas (hanya jika Ekuitas > 0)",
                "Profitabilitas bagi pemegang saham.", 0),
    FeatureSpec("net_margin", "Margin Laba Bersih", "Profitabilitas", "Laba Bersih / Pendapatan",
                "Margin laba yang tidak wajar dapat menandakan manajemen laba.", 0),
    FeatureSpec("gross_margin", "Margin Kotor", "Profitabilitas", "(Pendapatan - HPP) / Pendapatan",
                "Margin yang terlalu tinggi dapat menandakan HPP yang diturunkan.", 0),
    FeatureSpec("operating_margin", "Margin Operasi", "Profitabilitas", "EBIT / Pendapatan",
                "Profitabilitas inti usaha.", 0),
    FeatureSpec("ebit_to_assets", "EBIT terhadap Aset", "Profitabilitas", "EBIT / Total Aset",
                "Produktivitas aset sebelum bunga dan pajak (komponen X3 Altman).", -1),
    # --- Aktivitas ---------------------------------------------------------
    FeatureSpec("asset_turnover", "Perputaran Aset", "Aktivitas", "Pendapatan / Total Aset",
                "Efisiensi penggunaan aset (komponen X5 Altman).", 0),
    FeatureSpec("dso", "Hari Penagihan Piutang (DSO)", "Aktivitas", "Piutang / Pendapatan x 365",
                "Penagihan yang melambat adalah sinyal klasik pendapatan fiktif.", +1),
    FeatureSpec("dio", "Hari Persediaan (DIO)", "Aktivitas", "Persediaan / HPP x 365",
                "Persediaan menumpuk dapat menandakan penggelembungan nilai persediaan.", +1),
    # --- Struktur aset -----------------------------------------------------
    FeatureSpec("receivables_to_assets", "Piutang terhadap Aset", "Struktur Aset", "Piutang / Total Aset",
                "Porsi piutang yang besar lebih mudah dimanipulasi.", +1),
    FeatureSpec("inventory_to_assets", "Persediaan terhadap Aset", "Struktur Aset", "Persediaan / Total Aset",
                "Porsi persediaan yang besar lebih mudah dimanipulasi.", +1),
    FeatureSpec("ppe_to_assets", "Aset Tetap terhadap Aset", "Struktur Aset", "Aset Tetap Neto / Total Aset",
                "Intensitas modal.", 0),
    FeatureSpec("retained_earnings_to_assets", "Laba Ditahan terhadap Aset", "Struktur Aset",
                "Laba Ditahan / Total Aset", "Akumulasi laba (komponen X2 Altman); perusahaan muda lebih rentan.", -1),
    FeatureSpec("working_capital_to_assets", "Modal Kerja terhadap Aset", "Struktur Aset",
                "(Aset Lancar - Liabilitas Jangka Pendek) / Total Aset", "Komponen X1 Altman.", -1),
    # --- Pertumbuhan, kualitas laba, pasar, ukuran ---------------------------
    FeatureSpec("asset_growth", "Pertumbuhan Aset", "Pertumbuhan", "Total Aset t / Total Aset t-1 - 1",
                "Pertumbuhan aset yang tidak diikuti pendapatan patut diwaspadai.", +1),
    FeatureSpec("cfo_to_assets", "Arus Kas Operasi terhadap Aset", "Kualitas Laba",
                "Arus Kas Operasi / Total Aset", "Kas operasi yang lemah dibanding laba menandakan kualitas laba rendah.", -1),
    FeatureSpec("cfo_to_revenue", "Arus Kas Operasi terhadap Pendapatan", "Kualitas Laba",
                "Arus Kas Operasi / Pendapatan", "Konversi pendapatan menjadi kas.", -1),
    FeatureSpec("mve_to_liabilities", "Nilai Pasar Ekuitas terhadap Liabilitas", "Pasar",
                "Nilai Pasar Ekuitas / Total Liabilitas", "Komponen X4 Altman; penyangga terhadap kebangkrutan.", -1),
    FeatureSpec("market_to_book", "Market-to-Book", "Pasar", "Nilai Pasar Ekuitas / Ekuitas (hanya jika Ekuitas > 0)",
                "Ekspektasi pasar yang tinggi menambah tekanan memenuhi target.", +1),
    FeatureSpec("log_total_assets", "Ukuran Perusahaan (ln Aset)", "Ukuran", "ln(Total Aset)",
                "Variabel kontrol ukuran; transformasi log meredam kemiringan ekstrem.", 0),
    # --- Digit -------------------------------------------------------------
    FeatureSpec("round_share_1000", "Porsi Angka Bulat Ribuan", "Digit",
                "Porsi pos laporan yang berkelipatan 1000",
                "Digit preference: angka karangan cenderung dibulatkan (lihat uji digit terakhir).", +1),
]

FEATURE_NAMES: list[str] = [f.name for f in FEATURE_CATALOG]
BENEISH_INDEX_NAMES: list[str] = ["dsri", "gmi", "aqi", "sgi", "depi", "sgai", "lvgi", "tata"]


def catalog_as_records() -> list[dict]:
    """Katalog fitur sebagai list of dict (untuk API/frontend)."""
    return [asdict(f) for f in FEATURE_CATALOG]


def safe_divide(num: pd.Series | np.ndarray, den: pd.Series | np.ndarray, eps: float = 1e-9) -> pd.Series:
    """Pembagian aman: penyebut |den| <= eps -> NaN, hasil inf -> NaN."""
    num = pd.Series(num) if not isinstance(num, pd.Series) else num
    den = pd.Series(den, index=num.index) if not isinstance(den, pd.Series) else den
    out = num / den.where(den.abs() > eps)
    return out.replace([np.inf, -np.inf], np.nan)


def _numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in NUMERIC_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom berikut diperlukan untuk menghitung rasio: {', '.join(missing)}")
    return df[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce").astype(float)


def compute_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Hitung seluruh fitur pada ``FEATURE_CATALOG``. Indeks baris sama dengan ``df``.

    Kolom hasil berurutan sesuai ``FEATURE_NAMES``. Nilai tak terdefinisi = NaN.
    """
    d = _numeric_frame(df)
    dv = safe_divide
    gm_t = dv(d["revenue"] - d["cogs"], d["revenue"])
    gm_p = dv(d["revenue_prev"] - d["cogs_prev"], d["revenue_prev"])
    aq_t = 1.0 - dv(d["current_assets"] + d["ppe_net"], d["total_assets"])
    aq_p = 1.0 - dv(d["current_assets_prev"] + d["ppe_net_prev"], d["total_assets_prev"])
    dep_rate_t = dv(d["depreciation"], d["depreciation"] + d["ppe_net"])
    dep_rate_p = dv(d["depreciation_prev"], d["depreciation_prev"] + d["ppe_net_prev"])
    lev_t = dv(d["current_liabilities"] + d["long_term_debt"], d["total_assets"])
    lev_p = dv(d["current_liabilities_prev"] + d["long_term_debt_prev"], d["total_assets_prev"])
    pos_equity = d["equity"].where(d["equity"] > 0)

    f = pd.DataFrame(index=df.index)
    f["dsri"] = dv(dv(d["receivables"], d["revenue"]), dv(d["receivables_prev"], d["revenue_prev"]))
    f["gmi"] = dv(gm_p, gm_t)
    f["aqi"] = dv(aq_t, aq_p)
    f["sgi"] = dv(d["revenue"], d["revenue_prev"])
    f["depi"] = dv(dep_rate_p, dep_rate_t)
    f["sgai"] = dv(dv(d["sga"], d["revenue"]), dv(d["sga_prev"], d["revenue_prev"]))
    f["lvgi"] = dv(lev_t, lev_p)
    f["tata"] = dv(d["net_income"] - d["operating_cash_flow"], d["total_assets"])

    f["current_ratio"] = dv(d["current_assets"], d["current_liabilities"])
    f["quick_ratio"] = dv(d["current_assets"] - d["inventory"], d["current_liabilities"])
    f["cash_ratio"] = dv(d["cash"], d["current_liabilities"])

    f["debt_to_assets"] = dv(d["total_liabilities"], d["total_assets"])
    f["debt_to_equity"] = dv(d["total_liabilities"], pos_equity)
    f["long_term_debt_to_assets"] = dv(d["long_term_debt"], d["total_assets"])
    f["interest_coverage"] = dv(d["ebit"], d["interest_expense"])

    f["roa"] = dv(d["net_income"], d["total_assets"])
    f["roe"] = dv(d["net_income"], pos_equity)
    f["net_margin"] = dv(d["net_income"], d["revenue"])
    f["gross_margin"] = gm_t
    f["operating_margin"] = dv(d["ebit"], d["revenue"])
    f["ebit_to_assets"] = dv(d["ebit"], d["total_assets"])

    f["asset_turnover"] = dv(d["revenue"], d["total_assets"])
    f["dso"] = dv(d["receivables"], d["revenue"]) * 365.0
    f["dio"] = dv(d["inventory"], d["cogs"]) * 365.0

    f["receivables_to_assets"] = dv(d["receivables"], d["total_assets"])
    f["inventory_to_assets"] = dv(d["inventory"], d["total_assets"])
    f["ppe_to_assets"] = dv(d["ppe_net"], d["total_assets"])
    f["retained_earnings_to_assets"] = dv(d["retained_earnings"], d["total_assets"])
    f["working_capital_to_assets"] = dv(d["current_assets"] - d["current_liabilities"], d["total_assets"])

    f["asset_growth"] = dv(d["total_assets"], d["total_assets_prev"]) - 1.0
    f["cfo_to_assets"] = dv(d["operating_cash_flow"], d["total_assets"])
    f["cfo_to_revenue"] = dv(d["operating_cash_flow"], d["revenue"])
    f["mve_to_liabilities"] = dv(d["market_value_equity"], d["total_liabilities"])
    f["market_to_book"] = dv(d["market_value_equity"], pos_equity)
    f["log_total_assets"] = np.log(d["total_assets"].where(d["total_assets"] > 0))

    f["round_share_1000"] = rounding_features(df)["round_share_1000"]
    f = f.replace([np.inf, -np.inf], np.nan)
    return f[FEATURE_NAMES]


def beneish_m_score(indices: pd.DataFrame) -> pd.Series:
    """M-Score 8-variabel dari kolom dsri, gmi, aqi, sgi, depi, sgai, lvgi, tata."""
    c = BENEISH_COEF
    return (c["intercept"] + c["dsri"] * indices["dsri"] + c["gmi"] * indices["gmi"]
            + c["aqi"] * indices["aqi"] + c["sgi"] * indices["sgi"] + c["depi"] * indices["depi"]
            + c["sgai"] * indices["sgai"] + c["tata"] * indices["tata"] + c["lvgi"] * indices["lvgi"])


def _zone(z: pd.Series, distress_below: float, safe_above: float) -> pd.Series:
    out = pd.Series(np.where(z < distress_below, "Distress", np.where(z > safe_above, "Aman", "Abu-abu")),
                    index=z.index, dtype=object)
    out[z.isna()] = None
    return out


def compute_benchmark_scores(df: pd.DataFrame, m_threshold: float = BENEISH_THRESHOLD) -> pd.DataFrame:
    """Skor benchmark berbasis teori: Beneish M-Score dan Altman Z / Z''.

    Kolom: m_score, m_flag (1 jika M > ambang), altman_z, altman_z_zone,
    altman_z2, altman_z2_zone. Zona Z: Distress < 1,81 | Aman > 2,99.
    Zona Z'': Distress < 1,1 | Aman > 2,6.
    """
    d = _numeric_frame(df)
    ratios = compute_ratios(df)
    m = beneish_m_score(ratios)

    ta, tl = d["total_assets"], d["total_liabilities"]
    x1 = safe_divide(d["current_assets"] - d["current_liabilities"], ta)
    x2 = safe_divide(d["retained_earnings"], ta)
    x3 = safe_divide(d["ebit"], ta)
    x4_mkt = safe_divide(d["market_value_equity"], tl)
    x4_book = safe_divide(d["equity"], tl)
    x5 = safe_divide(d["revenue"], ta)

    c, c2 = ALTMAN_Z_COEF, ALTMAN_Z2_COEF
    z = c["x1"] * x1 + c["x2"] * x2 + c["x3"] * x3 + c["x4"] * x4_mkt + c["x5"] * x5
    z2 = c2["intercept"] + c2["x1"] * x1 + c2["x2"] * x2 + c2["x3"] * x3 + c2["x4"] * x4_book

    out = pd.DataFrame(index=df.index)
    out["m_score"] = m
    out["m_flag"] = np.where(m.isna(), np.nan, (m > m_threshold).astype(float))
    out["altman_z"] = z
    out["altman_z_zone"] = _zone(z, 1.81, 2.99)
    out["altman_z2"] = z2
    out["altman_z2_zone"] = _zone(z2, 1.1, 2.6)
    return out
