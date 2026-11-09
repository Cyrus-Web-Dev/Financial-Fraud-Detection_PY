"""Uji modul Benford. Jalankan: python -m tests.test_benford  (atau pytest)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.benford import (
    BENFORD_FIRST, benford_by_group, benford_first_digit_test, extract_first_digits,
    extract_last_digits, last_digit_test, rounding_features,
)


def test_benford_probabilities_sum_to_one() -> None:
    assert abs(BENFORD_FIRST.sum() - 1.0) < 1e-12
    assert abs(BENFORD_FIRST[0] - 0.30103) < 1e-4


def test_extract_first_digits_edge_cases() -> None:
    vals = [1, 9, 10, 99, 100, 1000, 0.5, 0.0123, -345, 1e9, 999.9999, 0, np.nan, np.inf]
    d = extract_first_digits(vals)
    assert d.tolist() == [1, 9, 1, 9, 1, 1, 5, 1, 3, 1, 9]


def test_first_digit_conforming_sample() -> None:
    rng = np.random.default_rng(0)
    x = 10 ** rng.uniform(0, 6, 20000)  # log-seragam -> memenuhi Benford
    r = benford_first_digit_test(x)
    assert r.mad < 0.006 and r.conformity == "Sangat sesuai"
    assert r.n == 20000 and r.dof == 8


def test_first_digit_nonconforming_sample() -> None:
    rng = np.random.default_rng(0)
    x = rng.uniform(1000, 9999, 5000)  # digit pertama seragam 1-9 -> melanggar Benford
    r = benford_first_digit_test(x)
    assert r.mad > 0.015 and r.conformity == "Tidak sesuai"
    assert r.p_value < 1e-6


def test_last_digit_uniform_vs_rounded() -> None:
    rng = np.random.default_rng(1)
    uniform = rng.integers(1000, 10**7, 20000)
    assert last_digit_test(uniform).p_value > 0.01
    rounded = uniform.copy(); rounded[:6000] = rounded[:6000] // 10 * 10
    r = last_digit_test(rounded)
    assert r.p_value < 1e-10 and "Tidak seragam" in r.conformity


def test_last_digit_ignores_small_and_noninteger() -> None:
    d = extract_last_digits([5, 99, 100, 101, 205.5, 1234, np.nan])
    assert d.tolist() == [0, 1, 4]


def test_empty_input_is_safe() -> None:
    r = benford_first_digit_test([])
    assert r.n == 0 and r.conformity == "Data tidak cukup" and r.small_sample


def test_by_group_on_sample() -> None:
    from src.config import SAMPLE_CLEAN_FILE
    df = pd.read_csv(SAMPLE_CLEAN_FILE)
    res = benford_by_group(df)
    assert set(res) == {"semua", "non_fraud", "fraud"}
    assert res["non_fraud"]["first_digit"]["conformity"] in {"Sangat sesuai", "Cukup sesuai"}
    # Baris fraud memiliki pembulatan ke ribuan -> digit terakhir tidak seragam
    assert res["fraud"]["last_digit"]["p_value"] < res["non_fraud"]["last_digit"]["p_value"]


def test_rounding_features() -> None:
    df = pd.DataFrame({"revenue": [5000, 1234, 500], "net_income": [2000, 5678, 10]})
    f = rounding_features(df, ["revenue", "net_income"])["round_share_1000"].tolist()
    assert f == [1.0, 0.0, 0.0]  # baris ke-3: tidak ada pos berukuran >= 1000


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"OK  {t.__name__}")
    print(f"\n{len(tests)} uji lulus.")
