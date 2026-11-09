"""Uji generator data sintetis.

Bisa dijalankan dengan pytest:        pytest tests -q
atau tanpa pytest (dari root proyek): python -m tests.test_synthetic
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.schema import ALL_COLUMNS, NUMERIC_COLUMNS, TARGET_COLUMN
from src.data.synthetic import generate_financial_panel, inject_data_issues


def _panel() -> pd.DataFrame:
    return generate_financial_panel(n_firms=400, years=(2020, 2021, 2022, 2023), seed=7)


def test_columns_and_shape() -> None:
    df = _panel()
    assert list(df.columns) == ALL_COLUMNS
    assert len(df) == 400 * 4
    assert df[NUMERIC_COLUMNS].notna().all().all()
    assert np.isfinite(df[NUMERIC_COLUMNS].to_numpy(dtype=float)).all()


def test_reproducible() -> None:
    a = generate_financial_panel(n_firms=50, years=(2021, 2022), seed=1)
    b = generate_financial_panel(n_firms=50, years=(2021, 2022), seed=1)
    c = generate_financial_panel(n_firms=50, years=(2021, 2022), seed=2)
    pd.testing.assert_frame_equal(a, b)
    assert not a["revenue"].equals(c["revenue"])


def test_balance_sheet_identities() -> None:
    df = _panel()
    assert (df["equity"] == df["total_assets"] - df["total_liabilities"]).all()
    assert (df["total_liabilities"] == df["current_liabilities"] + df["long_term_debt"]).all()
    assert (df["total_assets"] >= df["current_assets"] + df["ppe_net"]).all()
    assert (df["current_assets"] >= df["cash"] + df["receivables"] + df["inventory"]).all()


def test_retained_earnings_not_above_equity() -> None:
    df = _panel()
    pos = df["equity"] > 0
    assert (df.loc[pos, "retained_earnings"] <= df.loc[pos, "equity"]).all()


def test_positive_core_items() -> None:
    df = _panel()
    for col in ["total_assets", "revenue", "receivables", "ppe_net", "cogs", "sga",
                "total_assets_prev", "revenue_prev"]:
        assert (df[col] > 0).all(), col


def test_panel_consistency_prev_equals_last_year() -> None:
    """Nilai *_prev tahun t harus sama dengan nilai tahun t-1 perusahaan yang sama."""
    df = _panel().sort_values(["company_id", "tahun"])
    for cur_col, prev_col in [("total_assets", "total_assets_prev"),
                              ("revenue", "revenue_prev"),
                              ("receivables", "receivables_prev")]:
        shifted = df.groupby("company_id")[cur_col].shift(1)
        mask = shifted.notna()
        assert (shifted[mask] == df.loc[mask, prev_col]).all(), cur_col


def test_fraud_prevalence_and_target() -> None:
    df = generate_financial_panel(n_firms=1500, years=(2021, 2022, 2023), fraud_prevalence=0.06, seed=3)
    assert set(df[TARGET_COLUMN].unique()) <= {0, 1}
    rate = df[TARGET_COLUMN].mean()
    assert 0.04 < rate < 0.08, rate


def test_fraud_has_signal_but_overlaps() -> None:
    """Fraud harus punya sinyal (DSRI dan TATA lebih buruk) tetapi tidak terpisah sempurna."""
    df = generate_financial_panel(n_firms=1500, years=(2021, 2022, 2023), seed=11)
    dsri = (df["receivables"] / df["revenue"]) / (df["receivables_prev"] / df["revenue_prev"])
    tata = (df["net_income"] - df["operating_cash_flow"]) / df["total_assets"]
    fraud = df[TARGET_COLUMN] == 1
    assert dsri[fraud].median() > dsri[~fraud].median() * 1.05
    assert tata[fraud].median() > tata[~fraud].median()
    # tumpang tindih: ada fraud dengan DSRI di bawah median non-fraud
    assert (dsri[fraud] < dsri[~fraud].median()).any()


def test_digit_preference_in_fraud_rows() -> None:
    """Baris fraud lebih sering berakhiran 000 pada pendapatan dibanding non-fraud."""
    df = generate_financial_panel(n_firms=1500, years=(2021, 2022, 2023), seed=5)
    round_flag = (df["revenue"] % 1000 == 0)
    fraud = df[TARGET_COLUMN] == 1
    assert round_flag[fraud].mean() > round_flag[~fraud].mean() + 0.2


def test_inject_data_issues() -> None:
    df = _panel()
    dirty = inject_data_issues(df, missing_rate=0.02, outlier_rate=0.005, seed=1)
    assert dirty[NUMERIC_COLUMNS].isna().to_numpy().mean() > 0.01
    assert dirty[TARGET_COLUMN].equals(df[TARGET_COLUMN])
    assert dirty[["company_id", "tahun", "sektor"]].equals(df[["company_id", "tahun", "sektor"]])
    assert df[NUMERIC_COLUMNS].notna().all().all()  # data asli tidak termutasi


def test_invalid_arguments() -> None:
    for kwargs in [dict(n_firms=5), dict(fraud_prevalence=0.0), dict(fraud_prevalence=0.7),
                   dict(years=(2020, 2022))]:
        try:
            generate_financial_panel(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"ValueError tidak muncul untuk {kwargs}")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"OK  {t.__name__}")
    print(f"\n{len(tests)} uji lulus.")
