"""Penjelasan lokal: mengapa satu perusahaan diberi skor risiko tertentu."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.ratios import FEATURE_CATALOG


def explain_instance(phi_row: np.ndarray, x_row: pd.Series, base_value: float, names: list[str],
                     train_reference: pd.DataFrame | None = None, unit: str = "probabilitas", top_n: int = 5) -> dict:
    """Penjelasan satu observasi: prediksi = nilai dasar + jumlah kontribusi fitur (data untuk force plot).

    Parameters
    ----------
    phi_row : nilai Shapley satu observasi (urutan sama dengan ``names``).
    x_row : nilai fitur asli observasi (indeks = nama fitur).
    base_value : rata-rata prediksi pada sampel latar.
    train_reference : data latih untuk menghitung persentil nilai fitur (opsional).
    unit : "probabilitas" atau "log-odds" (untuk teks).
    """
    labels = {f.name: f.label for f in FEATURE_CATALOG}
    phi_row = np.asarray(phi_row, dtype=float)
    if len(phi_row) != len(names):
        raise ValueError("phi_row dan names harus sama panjang.")
    rows = []
    for j, n in enumerate(names):
        pct = float((train_reference[n] <= x_row[n]).mean() * 100) if train_reference is not None else np.nan
        rows.append({"fitur": n, "label": labels.get(n, n), "nilai": float(x_row[n]), "persentil_latih": pct,
                     "kontribusi": float(phi_row[j])})
    contrib = pd.DataFrame(rows).sort_values("kontribusi", key=np.abs, ascending=False).reset_index(drop=True)
    prediction = float(base_value + phi_row.sum())
    fmt = (lambda v: f"{v * 100:+.1f} poin persen".replace(".", ",")) if unit == "probabilitas" \
        else (lambda v: f"{v:+.2f} log-odds".replace(".", ","))
    sentences = []
    for _, r in contrib.head(top_n).iterrows():
        arah = "menaikkan" if r["kontribusi"] > 0 else "menurunkan"
        pos = f" (persentil ke-{r['persentil_latih']:.0f} dibanding data latih)" if np.isfinite(r["persentil_latih"]) else ""
        nilai = f"{r['nilai']:.3g}".replace(".", ",")
        sentences.append(f"{r['label']} bernilai {nilai}{pos} {arah} risiko ({fmt(r['kontribusi'])}).")
    return {"nilai_dasar": float(base_value), "prediksi": prediction, "kontribusi": contrib,
            "kalimat": sentences, "satuan": unit}
