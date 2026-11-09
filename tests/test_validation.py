"""Uji loader dan validator. Jalankan: python -m tests.test_validation"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

from src.config import SAMPLE_CLEAN_FILE, SAMPLE_DIRTY_FILE
from src.data.loader import DataLoadError, load_dataset, normalize_column_name
from src.data.validation import validate_dataset


def _clean() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_CLEAN_FILE)


def _codes(report) -> set[str]:
    return {i.code for i in report.issues}


# ----------------------------- loader --------------------------------------
def test_normalize_column_name() -> None:
    assert normalize_column_name(" Total Assets ") == "total_assets"
    assert normalize_column_name("Total-Assets") == "total_assets"


def test_load_csv_path_and_bytes_and_semicolon() -> None:
    df = load_dataset(SAMPLE_CLEAN_FILE)
    assert df.shape == (3600, 34)
    raw = SAMPLE_CLEAN_FILE.read_bytes()
    assert load_dataset(raw, filename="x.csv").shape == df.shape
    semi = df.head(5).to_csv(index=False, sep=";").encode("utf-8")
    assert load_dataset(semi, filename="semi.csv").shape == (5, 34)


def test_load_xlsx_and_header_normalization() -> None:
    df = _clean().head(10).rename(columns={"total_assets": "Total Assets", "revenue": "REVENUE"})
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    out = load_dataset(buf.getvalue(), filename="data.xlsx")
    assert "total_assets" in out.columns and "revenue" in out.columns and len(out) == 10


def test_load_errors() -> None:
    cases = [(b"", "a.csv"), (b"a,b\n1,2", "a.txt"), (b"a,b\n1,2", None), (b"a,b\n", "a.csv"),
             (b"a,a\n1,2", "a.csv")]
    for raw, name in cases:
        try:
            load_dataset(raw, filename=name)
        except DataLoadError:
            continue
        raise AssertionError(f"DataLoadError tidak muncul untuk {name!r}, {raw!r}")
    try:
        load_dataset(b"a,b\n1,2", filename="a.csv", max_mb=0)
    except DataLoadError as e:
        assert "batas" in str(e)
    else:
        raise AssertionError("batas ukuran tidak ditegakkan")
    try:
        load_dataset("/tidak/ada.csv")
    except DataLoadError:
        pass
    else:
        raise AssertionError("file tidak ada harus ditolak")


def test_latin1_encoding_fallback() -> None:
    raw = "company_id;sektor\nPRS-1;Jasa é\n".encode("latin-1")
    df = load_dataset(raw, filename="l.csv")
    assert df.shape == (1, 2)


# ----------------------------- validator -----------------------------------
def test_clean_sample_is_valid() -> None:
    df, rep = validate_dataset(_clean(), require_target=True)
    assert rep.is_valid, [i.message for i in rep.errors]
    assert rep.summary["n_baris"] == 3600 and rep.summary["n_perusahaan"] == 600
    assert rep.summary["rentang_tahun"] == [2018, 2023]
    assert "IMBALANCE" in _codes(rep)
    assert not ({"MISSING", "DUPLIKAT", "NERACA_TIDAK_SEIMBANG", "KONTINUITAS_PANEL"} & _codes(rep))


def test_dirty_sample_reports_missing_and_outliers_but_stays_valid() -> None:
    raw = pd.read_csv(SAMPLE_DIRTY_FILE)
    _, rep = validate_dataset(raw, require_target=True)
    assert rep.is_valid, [i.message for i in rep.errors]
    codes = _codes(rep)
    assert "MISSING" in codes and "OUTLIER" in codes
    assert rep.summary["jumlah_sel_outlier"] > 100
    assert rep.summary["baris_dengan_missing"] > 0


def test_missing_required_column() -> None:
    _, rep = validate_dataset(_clean().drop(columns=["receivables", "revenue"]))
    assert not rep.is_valid and "KOLOM_HILANG" in _codes(rep)
    assert "receivables" in rep.errors[0].message


def test_target_optional_for_prediction_but_required_for_training() -> None:
    df = _clean().drop(columns=["is_fraud"])
    _, rep = validate_dataset(df, require_target=False)
    assert rep.is_valid
    _, rep = validate_dataset(df, require_target=True)
    assert not rep.is_valid


def test_non_numeric_values() -> None:
    df = _clean()
    df["revenue"] = df["revenue"].astype(object)
    df.loc[3, "revenue"] = "1.234,5 jt"
    _, rep = validate_dataset(df)
    assert "BUKAN_ANGKA" in _codes(rep) and not rep.is_valid


def test_duplicates_and_nonpositive_denominators() -> None:
    df = pd.concat([_clean(), _clean().head(2)], ignore_index=True)
    _, rep = validate_dataset(df)
    assert "DUPLIKAT" in _codes(rep) and not rep.is_valid
    df = _clean(); df.loc[0, "total_assets"] = 0
    _, rep = validate_dataset(df)
    assert "PENYEBUT_NON_POSITIF" in _codes(rep) and not rep.is_valid


def test_balance_sheet_and_continuity_warnings() -> None:
    df = _clean()
    df.loc[10, "equity"] += df.loc[10, "total_assets"]           # neraca tidak seimbang
    df.loc[1000, "revenue_prev"] *= 2                             # t-1 tidak sama dengan tahun lalu
    _, rep = validate_dataset(df)
    codes = _codes(rep)
    assert "NERACA_TIDAK_SEIMBANG" in codes and "KONTINUITAS_PANEL" in codes
    assert rep.is_valid  # keduanya hanya peringatan


def test_invalid_target_values() -> None:
    df = _clean(); df.loc[0, "is_fraud"] = 2
    _, rep = validate_dataset(df, require_target=True)
    assert "LABEL_TIDAK_BINER" in _codes(rep)
    df = _clean(); df["is_fraud"] = 0
    _, rep = validate_dataset(df, require_target=True)
    assert "LABEL_SATU_KELAS" in _codes(rep)
    df = _clean(); df["is_fraud"] = 0; df.loc[:9, "is_fraud"] = 1
    _, rep = validate_dataset(df, require_target=True)
    assert "FRAUD_SEDIKIT" in _codes(rep)


def test_invalid_year_and_empty() -> None:
    df = _clean(); df["tahun"] = df["tahun"].astype(float); df.loc[0, "tahun"] = 2018.5
    _, rep = validate_dataset(df)
    assert "TAHUN_TIDAK_VALID" in _codes(rep)
    _, rep = validate_dataset(_clean().head(0))
    assert "DATA_KOSONG" in _codes(rep)


def test_column_with_too_much_missing_is_error() -> None:
    df = _clean(); df.loc[: int(len(df) * 0.5), "inventory"] = np.nan
    _, rep = validate_dataset(df)
    assert any(i.code == "MISSING" and i.severity == "error" and i.column == "inventory" for i in rep.issues)


def test_input_not_mutated_and_report_serializable() -> None:
    import json
    raw = _clean(); snapshot = raw.copy(deep=True)
    clean, rep = validate_dataset(raw, require_target=True)
    pd.testing.assert_frame_equal(raw, snapshot)
    json.dumps(rep.to_dict(), allow_nan=False)
    assert clean["tahun"].dtype.kind == "i"


def test_end_to_end_load_validate() -> None:
    df = load_dataset(SAMPLE_DIRTY_FILE)
    _, rep = validate_dataset(df, require_target=True)
    assert rep.is_valid


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"OK  {t.__name__}")
    print(f"\n{len(tests)} uji lulus.")
