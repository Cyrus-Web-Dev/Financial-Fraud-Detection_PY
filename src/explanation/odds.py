"""Interpretasi odds ratio regresi logistik dan penghubungannya dengan teori keuangan.

Odds ratio (OR) per 1 simpangan baku (SD) fitur: bila OR = 2 maka kenaikan 1 SD melipatgandakan odds
fraud, dengan fitur lain tetap (ceteris paribus). Bila OR < 1, odds fraud turun sebesar (1 - OR) x 100%.
Arah yang diamati dibandingkan dengan arah yang diharapkan teori pada ``FEATURE_CATALOG``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.ratios import FEATURE_CATALOG

ALPHA = 0.05


def _id(x: float, nd: int = 2) -> str:
    """Format angka berkoma desimal (gaya Indonesia)."""
    return f"{x:.{nd}f}".replace(".", ",")


def _p_text(p: float) -> str:
    return "p < 0,001" if p < 0.001 else f"p = {_id(p, 3)}"


def interpret_odds_ratios(summary: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    """Tabel interpretasi untuk tiap fitur (intersep dilewati).

    ``summary`` berasal dari ``LogitInference.summary()`` (indeks = nama variabel) pada fitur terstandardisasi.
    Kolom keluaran: arah teramati, arah teori, kesesuaian, signifikansi, dan kalimat interpretasi.
    """
    cat = {f.name: f for f in FEATURE_CATALOG}
    rows = []
    for name, r in summary.iterrows():
        if name == "const":
            continue
        spec = cat.get(name)
        label = spec.label if spec else name
        or_, lo, hi, p = float(r["odds_ratio"]), float(r["or_ci_bawah"]), float(r["or_ci_atas"]), float(r["p_value"])
        sig = p < alpha
        observed = 1 if or_ > 1 else -1
        expected = spec.expected_sign if spec else 0
        if expected == 0:
            match = "arah tidak ditetapkan teori"
        elif not sig:
            match = "tidak signifikan, arah tidak dapat disimpulkan"
        else:
            match = "sesuai teori" if observed == expected else "BERLAWANAN dengan teori"
        if sig:
            effect = (f"meningkatkan odds fraud {_id(or_)} kali" if or_ > 1
                      else f"menurunkan odds fraud sebesar {_id((1 - or_) * 100, 0)}%")
            text = (f"Kenaikan 1 SD {label} {effect} (OR = {_id(or_)}; CI 95% {_id(lo)}-{_id(hi)}; {_p_text(p)}). "
                    f"Hasil ini {match}.")
        elif p < 0.10:
            arah = "positif" if or_ > 1 else "negatif"
            text = (f"{label} menunjukkan kecenderungan {arah} yang hanya marginal dan tidak signifikan pada "
                    f"alpha = {_id(alpha * 100, 0)}% (OR = {_id(or_)}; CI 95% {_id(lo)}-{_id(hi)}; {_p_text(p)}); "
                    f"bukti belum cukup kuat.")
        else:
            text = (f"{label} tidak berpengaruh signifikan terhadap odds fraud (OR = {_id(or_)}; "
                    f"CI 95% {_id(lo)}-{_id(hi)} mencakup 1; {_p_text(p)}).")
        rows.append({"fitur": name, "label": label, "odds_ratio": or_, "ci_bawah": lo, "ci_atas": hi, "p_value": p,
                     "signifikan": sig, "arah_teramati": "+" if observed > 0 else "-",
                     "arah_teori": {1: "+", -1: "-", 0: "?"}[expected], "kesesuaian": match,
                     "teori": spec.rationale if spec else "", "interpretasi": text})
    return pd.DataFrame(rows)


def linear_contributions(coef: pd.Series, z_row: pd.Series, z_mean: pd.Series | None = None) -> pd.DataFrame:
    """Kontribusi log-odds tiap fitur untuk satu observasi pada model linear: b_j (z_j - rata2_j).

    Dengan ``z_mean`` = rata-rata latar; jika None dianggap 0 (fitur terstandardisasi, rata-rata latih = 0).
    """
    names = [n for n in coef.index if n != "const"]
    mean = z_mean.reindex(names) if z_mean is not None else pd.Series(0.0, index=names)
    contrib = coef[names] * (z_row[names] - mean)
    return pd.DataFrame({"fitur": names, "nilai_z": z_row[names].to_numpy(), "kontribusi_logodds": contrib.to_numpy()}
                        ).sort_values("kontribusi_logodds", key=np.abs, ascending=False).reset_index(drop=True)
