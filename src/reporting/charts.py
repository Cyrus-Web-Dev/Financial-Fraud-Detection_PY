"""Grafik untuk laporan PDF (matplotlib, backend Agg). Setiap fungsi mengembalikan (png_bytes, lebar_px, tinggi_px)."""
from __future__ import annotations

import io
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.reporting.fmt import model_label  # noqa: E402

NAVY, RED, BLUE, AMBER, GREEN, GREY = "#12305a", "#d7263d", "#2f6fde", "#e08a14", "#1a9a68", "#9aa8c0"
PALETTE = [NAVY, RED, BLUE, AMBER, GREEN, "#7a5cc4"]
DPI = 200

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c9d3e3", "axes.labelcolor": "#33425e", "xtick.color": "#5b6b85", "ytick.color": "#5b6b85",
    "axes.grid": True, "grid.color": "#eaeff7", "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.facecolor": "white",
})
Result = tuple[bytes, int, int]


def _png(fig: Any) -> Result:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight", facecolor="white")
    w, h = fig.get_size_inches() * DPI
    plt.close(fig)
    data = buf.getvalue()
    # ukuran sebenarnya setelah bbox_inches="tight" dibaca dari header PNG
    iw, ih = int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    return data, iw or int(w), ih or int(h)


def _pct(ax: Any, axis: str = "y") -> None:
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 100:.0f}%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def roc_pr_figure(kurva: dict[str, Any], models: Sequence[str]) -> Result:
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 3.1))
    for i, m in enumerate(models):
        c = kurva.get(m)
        if not c:
            continue
        col = PALETTE[i % len(PALETTE)]
        a.plot(c["roc"]["fpr"], c["roc"]["tpr"], color=col, lw=1.8, label=model_label(m))
        b.plot(c["pr"]["recall"], c["pr"]["precision"], color=col, lw=1.8, label=model_label(m))
    a.plot([0, 1], [0, 1], ls="--", color=GREY, lw=1)
    a.set(xlabel="Laju false positive", ylabel="Laju true positive (recall)", title="Kurva ROC", xlim=(0, 1), ylim=(0, 1.02))
    b.set(xlabel="Recall", ylabel="Precision", title="Kurva Precision-Recall", xlim=(0, 1), ylim=(0, 1.02))
    a.legend(fontsize=7, loc="lower right", frameon=False)
    fig.tight_layout()
    return _png(fig)


def calibration_figure(bins: Sequence[dict[str, Any]], title: str) -> Result:
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    x = np.array([b["rata_prediksi"] for b in bins]); y = np.array([b["frekuensi_teramati"] for b in bins])
    lo = np.array([b["ci95_bawah"] for b in bins]); hi = np.array([b["ci95_atas"] for b in bins])
    top = max(0.05, float(max(x.max(initial=0), hi.max(initial=0))) * 1.1)
    ax.plot([0, top], [0, top], ls="--", color=GREY, lw=1)
    ax.errorbar(x, y, yerr=[np.clip(y - lo, 0, None), np.clip(hi - y, 0, None)], fmt="o-", color=NAVY, ecolor=RED, lw=1.6, ms=4, capsize=2)
    ax.set(xlim=(0, top), ylim=(0, top), xlabel="Probabilitas prediksi", ylabel="Frekuensi fraud teramati", title=title)
    _pct(ax, "x"); _pct(ax, "y")
    fig.tight_layout()
    return _png(fig)


def metric_ci_figure(rows: Sequence[dict[str, Any]], metric: str, models: Sequence[str], title: str) -> Result:
    sel = [next((r for r in rows if r["model"] == m and r["metrik"] == metric), None) for m in models]
    pairs = [(m, r) for m, r in zip(models, sel) if r]
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    names = [model_label(m).replace(" (inferensial)", "").replace(" (L2)", " L2") for m, _ in pairs]
    est = np.array([r["estimasi"] for _, r in pairs])
    err = np.array([[r["estimasi"] - r["ci_bawah"] for _, r in pairs], [r["ci_atas"] - r["estimasi"] for _, r in pairs]])
    ax.bar(range(len(pairs)), est, color=BLUE, alpha=0.9, yerr=np.clip(err, 0, None), error_kw={"ecolor": "#0f1b33", "lw": 1.2, "capsize": 3})
    ax.set_xticks(range(len(pairs)), names, rotation=35, ha="right", fontsize=7.5)
    ax.set(ylim=(0, 1.05), title=title)
    fig.tight_layout()
    return _png(fig)


def hbar_figure(labels: Sequence[str], values: Sequence[float], title: str, xlabel: str, color: str = NAVY, signed: bool = False) -> Result:
    n = len(labels)
    fig, ax = plt.subplots(figsize=(6.6, max(1.6, 0.34 * n + 0.9)))
    y = np.arange(n)[::-1]
    cols = [(RED if v >= 0 else GREEN) for v in values] if signed else color
    ax.barh(y, values, color=cols, height=0.62)
    ax.set_yticks(y, [l if len(l) <= 40 else l[:39] + "…" for l in labels], fontsize=7.8)
    ax.set(title=title, xlabel=xlabel)
    ax.grid(axis="y", visible=False)
    if signed:
        ax.axvline(0, color="#7b8aa5", lw=0.8)
    fig.tight_layout()
    return _png(fig)


def forest_figure(rows: Sequence[dict[str, Any]], title: str) -> Result:
    n = len(rows)
    fig, ax = plt.subplots(figsize=(6.6, max(1.8, 0.36 * n + 0.9)))
    y = np.arange(n)[::-1]
    for yi, r in zip(y, rows):
        col = NAVY if r["signifikan"] else GREY
        ax.plot([max(r["ci_bawah"], 1e-3), r["ci_atas"]], [yi, yi], color=col, lw=2)
        ax.plot(r["odds_ratio"], yi, "D", color=col, ms=5)
    ax.axvline(1, color=RED, ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 4, 8], ["0,5", "1", "2", "4", "8"])
    ax.set_yticks(y, [r["label"] if len(r["label"]) <= 38 else r["label"][:37] + "…" for r in rows], fontsize=7.8)
    ax.set(title=title, xlabel="Odds ratio per 1 SD (skala log); titik gelap = signifikan")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _png(fig)


def histogram_figure(probs: Sequence[float], threshold: float) -> Result:
    edges = [0, 0.02, 0.05, 0.1, 0.2, 0.4, 1.0000001]
    labels = ["< 2%", "2–5%", "5–10%", "10–20%", "20–40%", "≥ 40%"]
    counts, _ = np.histogram(np.asarray(probs, dtype=float), bins=edges)
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    bars = ax.bar(range(6), counts, color=BLUE, alpha=0.9)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{c}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(range(6), labels)
    ax.set(ylabel="Jumlah observasi", title=f"Sebaran probabilitas terkalibrasi (ambang {threshold * 100:.1f}%)".replace(".", ","))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return _png(fig)


def benford_figure(first: dict[str, Any], title: str) -> Result:
    d = first["digits"]
    obs = [v if v is not None else 0 for v in first["observed_prop"]]
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.bar(d, obs, color=RED, alpha=0.75, label="Teramati")
    ax.plot(d, first["expected_prop"], "o--", color=NAVY, lw=1.5, ms=4, label="Harapan Benford")
    ax.set_xticks(d)
    ax.set(xlabel="Digit pertama", title=title)
    _pct(ax, "y")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return _png(fig)
