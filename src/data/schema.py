"""Skema kolom data mentah laporan keuangan.

Data asli (mis. dari IDX) harus dipetakan ke nama kolom di bawah ini.
Satuan nilai uang: Rupiah juta (konsisten di seluruh kolom).

Kolom *_prev adalah nilai tahun sebelumnya (t-1) dari perusahaan yang sama,
dibutuhkan untuk indeks Beneish M-Score dan rasio pertumbuhan.
"""
from __future__ import annotations

TARGET_COLUMN: str = "is_fraud"

IDENTIFIER_COLUMNS: list[str] = ["company_id", "tahun", "sektor"]

# Pos laporan keuangan tahun berjalan (t)
CURRENT_COLUMNS: list[str] = [
    "total_assets",
    "current_assets",
    "cash",
    "receivables",
    "inventory",
    "ppe_net",
    "current_liabilities",
    "long_term_debt",
    "total_liabilities",
    "equity",
    "retained_earnings",
    "revenue",
    "cogs",
    "sga",
    "depreciation",
    "ebit",
    "interest_expense",
    "net_income",
    "operating_cash_flow",
    "market_value_equity",
]

# Pos tahun sebelumnya (t-1) yang dibutuhkan untuk indeks Beneish
PREVIOUS_BASE_COLUMNS: list[str] = [
    "total_assets",
    "current_assets",
    "receivables",
    "ppe_net",
    "current_liabilities",
    "long_term_debt",
    "revenue",
    "cogs",
    "sga",
    "depreciation",
]
PREVIOUS_COLUMNS: list[str] = [f"{c}_prev" for c in PREVIOUS_BASE_COLUMNS]

NUMERIC_COLUMNS: list[str] = CURRENT_COLUMNS + PREVIOUS_COLUMNS

# Urutan kolom lengkap pada file data (label di paling akhir)
ALL_COLUMNS: list[str] = (
    IDENTIFIER_COLUMNS + CURRENT_COLUMNS + PREVIOUS_COLUMNS + [TARGET_COLUMN]
)

# Kolom wajib untuk data yang akan diprediksi (label tidak wajib)
REQUIRED_COLUMNS_FOR_PREDICTION: list[str] = IDENTIFIER_COLUMNS + NUMERIC_COLUMNS

# Label Bahasa Indonesia untuk tampilan UI
COLUMN_LABELS_ID: dict[str, str] = {
    "company_id": "ID Perusahaan",
    "tahun": "Tahun",
    "sektor": "Sektor",
    "total_assets": "Total Aset",
    "current_assets": "Aset Lancar",
    "cash": "Kas dan Setara Kas",
    "receivables": "Piutang Usaha",
    "inventory": "Persediaan",
    "ppe_net": "Aset Tetap Neto",
    "current_liabilities": "Liabilitas Jangka Pendek",
    "long_term_debt": "Liabilitas Jangka Panjang",
    "total_liabilities": "Total Liabilitas",
    "equity": "Ekuitas",
    "retained_earnings": "Laba Ditahan",
    "revenue": "Pendapatan",
    "cogs": "Beban Pokok Penjualan",
    "sga": "Beban Penjualan, Umum dan Administrasi",
    "depreciation": "Beban Penyusutan",
    "ebit": "Laba Sebelum Bunga dan Pajak (EBIT)",
    "interest_expense": "Beban Bunga",
    "net_income": "Laba Bersih",
    "operating_cash_flow": "Arus Kas Operasi",
    "market_value_equity": "Nilai Pasar Ekuitas",
    "is_fraud": "Label Fraud (1 = fraud)",
}
for _base in PREVIOUS_BASE_COLUMNS:
    COLUMN_LABELS_ID[f"{_base}_prev"] = f"{COLUMN_LABELS_ID[_base]} (t-1)"
