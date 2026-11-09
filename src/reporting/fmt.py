"""Pemformat angka bergaya Indonesia untuk laporan (titik ribuan, koma desimal)."""
from __future__ import annotations

import math
from typing import Any

DASH = "–"


def _ok(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def fnum(x: Any, digits: int = 2) -> str:
    """1234.5 -> '1.234,50'. Nilai tak terdefinisi -> '–'."""
    if not _ok(x):
        return DASH
    s = f"{x:,.{digits}f}"
    return s.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def fint(x: Any) -> str:
    return fnum(round(x), 0) if _ok(x) else DASH


def fpct(x: Any, digits: int = 1) -> str:
    """Proporsi 0-1 -> '6,1%'."""
    return f"{fnum(x * 100, digits)}%" if _ok(x) else DASH


def fprob(p: Any) -> str:
    if not _ok(p):
        return DASH
    if 0 < p < 0.001:
        return "< 0,1%"
    return fpct(p, 2 if p < 0.01 else 1)


def fp(p: Any) -> str:
    if not _ok(p):
        return DASH
    return "< 0,001" if p < 0.001 else fnum(p, 3)


def fsmart(x: Any) -> str:
    if not _ok(x):
        return DASH
    a = abs(x)
    if a == 0:
        return "0"
    if a >= 1000:
        return fint(x)
    if a >= 100:
        return fnum(x, 1)
    if a >= 1:
        return fnum(x, 2)
    if a >= 0.01:
        return fnum(x, 3)
    return f"{x:.1e}".replace(".", ",")


MODEL_LABELS = {
    "logit_mle": "Regresi Logistik (inferensial)", "logreg": "Regresi Logistik (L2)", "random_forest": "Random Forest",
    "hist_gb": "Gradient Boosting", "isolation_forest": "Isolation Forest", "m_score": "Beneish M-Score",
}


def model_label(name: str) -> str:
    return MODEL_LABELS.get(name, name)
