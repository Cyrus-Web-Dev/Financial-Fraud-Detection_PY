"""Pembangun laporan PDF (ReportLab). Menerima ``ReportData`` (dict JSON-aman) dan mengembalikan bytes PDF.

Setiap bagian dilewati bila datanya tidak tersedia, sehingga laporan tetap valid walaupun hasil
evaluasi/penjelasan belum dibuat. Font DejaVu Sans disertakan agar karakter matematis (≥, χ², →) tampil
benar di semua sistem operasi.
"""
from __future__ import annotations

import io
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, KeepTogether, NextPageTemplate, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle,
)

from src.reporting import charts
from src.reporting.fmt import DASH, fint, fnum, fp, fpct, fprob, fsmart, model_label

FONT_DIR = Path(__file__).resolve().parent / "fonts"
NAVY = colors.HexColor("#12305a")
NAVY_DARK = colors.HexColor("#0b1f3a")
RED = colors.HexColor("#d7263d")
LINE = colors.HexColor("#dde5f1")
ZEBRA = colors.HexColor("#f4f7fc")
MUTED = colors.HexColor("#5b6b85")
PAGE_W, PAGE_H = A4
MARGIN = 2.0 * cm
TEXT_W = PAGE_W - 2 * MARGIN
SUPERVISED = ("logreg", "random_forest", "hist_gb")


def _register_fonts() -> tuple[str, str, str]:
    """Daftarkan DejaVu Sans; jika berkas tidak ada, jatuh ke Helvetica (karakter khusus mungkin tidak tampil)."""
    regular, bold, italic = (FONT_DIR / n for n in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans-Oblique.ttf"))
    if all(p.exists() for p in (regular, bold, italic)):
        for name, path in (("DejaVu", regular), ("DejaVu-Bold", bold), ("DejaVu-Oblique", italic)):
            if name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(name, str(path)))
        pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu-Oblique", boldItalic="DejaVu-Bold")
        return "DejaVu", "DejaVu-Bold", "DejaVu-Oblique"
    return "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"


@dataclass
class ReportData:
    dataset: dict[str, Any]                 # nama, n_baris, punya_label, validasi{...}
    prediction: dict[str, Any]              # keluaran PredictionService.predict_frame
    eda: dict[str, Any] | None = None
    evaluation: dict[str, Any] | None = None
    explanation: dict[str, Any] | None = None
    modeling: dict[str, Any] | None = None
    model_info: dict[str, Any] | None = None
    top_n: int = 20
    synthetic: bool = False
    generated_at: str = ""
    title: str = "Laporan Analisis Indikasi Fraud Laporan Keuangan"
    notes: list[str] = field(default_factory=list)
    scope: str = ""                         # mis. 'tahun 2023' bila prediksi hanya untuk sebagian data


class _Styles:
    def __init__(self) -> None:
        f, fb, fi = _register_fonts()
        self.font, self.bold, self.italic = f, fb, fi
        base = dict(fontName=f, textColor=colors.HexColor("#0f1b33"), alignment=TA_LEFT)
        self.body = ParagraphStyle("body", fontSize=9.4, leading=13.6, spaceAfter=6, **base)
        self.small = ParagraphStyle("small", parent=self.body, fontSize=8, leading=11, textColor=MUTED, spaceAfter=4)
        self.h1 = ParagraphStyle("h1", parent=self.body, fontName=fb, fontSize=15, leading=19, textColor=NAVY, spaceBefore=6, spaceAfter=8)
        self.h2 = ParagraphStyle("h2", parent=self.body, fontName=fb, fontSize=11, leading=14, textColor=NAVY, spaceBefore=8, spaceAfter=4)
        self.cell = ParagraphStyle("cell", parent=self.body, fontSize=7.8, leading=9.6, spaceAfter=0)
        self.cell_r = ParagraphStyle("cell_r", parent=self.cell, alignment=2)
        self.head = ParagraphStyle("head", parent=self.cell, fontName=fb, textColor=colors.white)
        self.head_r = ParagraphStyle("head_r", parent=self.head, alignment=2)
        self.caption = ParagraphStyle("caption", parent=self.small, fontName=fi, spaceAfter=8)
        self.bullet = ParagraphStyle("bullet", parent=self.body, leftIndent=12, bulletIndent=2, spaceAfter=3)
        self.cover_title = ParagraphStyle("ct", parent=self.body, fontName=fb, fontSize=26, leading=31, textColor=colors.white)
        self.cover_sub = ParagraphStyle("cs", parent=self.body, fontSize=12, leading=16, textColor=colors.HexColor("#c3d3f0"))


def _p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(str(text)), style)


def _rich(text: str, style: ParagraphStyle) -> Paragraph:
    """Paragraf dengan tag <b>/<i> yang saya tulis sendiri; sisipan data harus sudah di-escape."""
    return Paragraph(text, style)


class _Builder:
    def __init__(self, data: ReportData) -> None:
        self.d = data
        self.s = _Styles()
        self.story: list[Any] = []
        self.model = data.prediction.get("model", "")
        self.label = model_label(self.model)
        self.thr = float(data.prediction.get("ambang", 0.0))
        self.section_no = 0

    # ------------------------------------------------------------- komponen
    def h1(self, t: str) -> None:
        """Judul bagian dengan nomor otomatis (nomor tidak melompat saat bagian opsional dilewati)."""
        self.section_no += 1
        self.story.append(_p(f"{self.section_no}. {t}", self.s.h1))

    def h2(self, t: str) -> None:
        self.story.append(_p(t, self.s.h2))

    def para(self, t: str) -> None:
        self.story.append(_p(t, self.s.body))

    def small(self, t: str) -> None:
        self.story.append(_p(t, self.s.small))

    def bullets(self, items: Sequence[str]) -> None:
        for it in items:
            self.story.append(Paragraph(escape(it), self.s.bullet, bulletText="•"))

    def figure(self, fig: tuple[bytes, int, int], width_cm: float, caption: str | None = None) -> Any:
        png, w, h = fig
        wp = width_cm * cm
        img = Image(io.BytesIO(png), width=wp, height=wp * h / w)
        return KeepTogether([img, _p(caption, self.s.caption)]) if caption else img

    def add_figure(self, fig: tuple[bytes, int, int], width_cm: float, caption: str | None = None) -> None:
        self.story.append(self.figure(fig, width_cm, caption))

    def side_by_side(self, a: Any, b: Any) -> None:
        t = Table([[a, b]], colWidths=[TEXT_W / 2, TEXT_W / 2])
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]))
        self.story.append(t)

    def table(self, headers: Sequence[str], rows: Sequence[Sequence[str]], widths: Sequence[float], right: Sequence[int] = ()) -> None:
        if not rows:
            self.small("Tidak ada data untuk ditampilkan.")
            return
        head = [_p(h, self.s.head_r if i in right else self.s.head) for i, h in enumerate(headers)]
        body = [[_p(c, self.s.cell_r if i in right else self.s.cell) for i, c in enumerate(r)] for r in rows]
        total = sum(widths)
        t = Table([head] + body, colWidths=[w / total * TEXT_W for w in widths], repeatRows=1)
        st = [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
              ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3), ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
        st += [("BACKGROUND", (0, i), (-1, i), ZEBRA) for i in range(2, len(body) + 1, 2)]
        t.setStyle(TableStyle(st))
        self.story.append(t)
        self.story.append(Spacer(1, 6))

    def kv(self, items: Sequence[tuple[str, str]]) -> None:
        self.table(["Keterangan", "Nilai"], [[k, v] for k, v in items], [1.1, 2.9])

    # ------------------------------------------------------------- data turunan
    @property
    def rows(self) -> list[dict[str, Any]]:
        return self.d.prediction.get("hasil", [])

    def _model_boot(self, metric: str) -> dict[str, Any] | None:
        ev = self.d.evaluation or {}
        return next((r for r in ev.get("metrik_uji_bootstrap", []) if r["model"] == self.model and r["metrik"] == metric), None)

    def _cost_row(self) -> dict[str, Any] | None:
        ev = self.d.evaluation or {}
        return next((r for r in ev.get("ambang_keputusan", []) if r["model"] == self.model and str(r["aturan"]).startswith("biaya_")), None)

    # ------------------------------------------------------------- bagian
    def cover(self) -> None:
        s = self.s
        self.story.append(Spacer(1, 5.2 * cm))
        self.story.append(_p(self.d.title, s.cover_title))
        self.story.append(Spacer(1, 10))
        self.story.append(_p(f"Dataset: {self.d.dataset.get('nama', DASH)}", s.cover_sub))
        self.story.append(Spacer(1, 4.2 * cm))
        info = [("Tanggal pembuatan", self.d.generated_at or DASH), ("Model prediksi", self.label), ("Ambang keputusan", fpct(self.thr, 1)),
                ("Jumlah observasi", fint(self.d.dataset.get("n_baris")))]
        t = Table([[_p(k, ParagraphStyle("k", fontName=s.font, fontSize=9, textColor=colors.HexColor("#8fa6cc"))),
                    _p(v, ParagraphStyle("v", fontName=s.bold, fontSize=10, textColor=colors.white))] for k, v in info], colWidths=[4.2 * cm, 10 * cm])
        t.setStyle(TableStyle([("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
        self.story.append(t)
        self.story.append(Spacer(1, 2.2 * cm))
        warn = "Skor risiko adalah indikasi untuk diperiksa lebih lanjut, bukan bukti fraud."
        if self.d.synthetic:
            warn += " Dataset ini SINTETIS (untuk pengembangan dan demo), sehingga hasil tidak boleh dipakai sebagai temuan substantif."
        self.story.append(_p(warn, ParagraphStyle("w", fontName=s.italic, fontSize=9, leading=13, textColor=colors.HexColor("#ffb3bd"))))
        self.story.append(NextPageTemplate("body"))
        self.story.append(PageBreak())

    def summary(self) -> None:
        d, p = self.d, self.d.prediction
        v = d.dataset.get("validasi", {}).get("summary", {})
        years = v.get("rentang_tahun")
        flagged, total = p.get("n_ditandai", 0), p.get("n_baris", 0)
        risk = defaultdict(int)
        for r in self.rows:
            risk[r["tingkat_risiko"]] += 1
        self.h1("Ringkasan Eksekutif")
        txt = (f"Laporan ini merangkum analisis terhadap {fint(d.dataset.get('n_baris'))} observasi laporan keuangan"
               + (f" dari {fint(v.get('n_perusahaan'))} perusahaan" if v.get("n_perusahaan") else "")
               + (f" (tahun {years[0]}–{years[1]})" if years else "")
               + f" menggunakan model {self.label}. Pada ambang keputusan {fpct(self.thr, 1)} (probabilitas terkalibrasi), sistem menandai "
               f"{fint(flagged)} dari {fint(total)} observasi ({fpct(flagged / total if total else 0, 1)}) untuk diperiksa lebih lanjut: "
               f"{fint(risk['Tinggi'])} berisiko tinggi dan {fint(risk['Sedang'])} berisiko sedang.")
        self.para(txt)
        if d.scope and total != d.dataset.get("n_baris"):
            self.para(f"Catatan cakupan: prediksi dijalankan pada {fint(total)} dari {fint(d.dataset.get('n_baris'))} observasi ({d.scope}).")
        eda_dist = (d.eda or {}).get("distribusi_kelas")
        if d.dataset.get("punya_label") and eda_dist:
            a = eda_dist[0]
            self.para(f"Dalam data terdapat {fint(a['fraud'])} observasi berlabel fraud ({fpct(a['proporsi_fraud'], 2)}); kelas fraud sangat jarang, "
                      "sehingga accuracy tidak dipakai sebagai ukuran utama.")
        pr, roc, cost = self._model_boot("pr_auc"), self._model_boot("roc_auc"), self._cost_row()
        if pr and roc:
            sent = (f"Pada data uji hold-out (perusahaan yang tidak pernah dilihat model), model ini memiliki PR-AUC {fnum(pr['estimasi'], 3)} "
                    f"(CI 95% {fnum(pr['ci_bawah'], 3)}–{fnum(pr['ci_atas'], 3)}) dan ROC-AUC {fnum(roc['estimasi'], 3)}.")
            if cost and cost.get("penghematan_biaya_persen") is not None:
                ev = d.evaluation or {}
                b = ev.get("biaya", {})
                sent += (f" Dengan asumsi biaya false negative : false positive = {fnum(b.get('c_fn'), 0)} : {fnum(b.get('c_fp'), 0)}, penggunaan model pada ambang "
                         f"berbasis biaya menghemat sekitar {fnum(cost['penghematan_biaya_persen'], 0)}% dibanding tanpa model.")
            self.para(sent + " Metrik ini berasal dari data uji tersendiri, bukan dari dataset yang dianalisis pada laporan ini.")
        self.h2("Lima observasi dengan probabilitas tertinggi")
        top = sorted(self.rows, key=lambda r: -r["probabilitas"])[:5]
        self.table(["Perusahaan", "Tahun", "Sektor", "Probabilitas", "Risiko"], [[r["company_id"], str(r["tahun"]), r["sektor"], fprob(r["probabilitas"]), r["tingkat_risiko"]] for r in top],
                   [1.3, 0.8, 1.5, 1.2, 1.0], right=(3,))

    def data_section(self) -> None:
        val = self.d.dataset.get("validasi", {})
        sm = val.get("summary", {})
        self.h1("Data dan Validasi")
        years = sm.get("rentang_tahun")
        self.kv([("Berkas", str(self.d.dataset.get("nama", DASH))), ("Jumlah baris / kolom", f"{fint(sm.get('n_baris'))} / {fint(sm.get('n_kolom'))}"),
                 ("Jumlah perusahaan", fint(sm.get("n_perusahaan"))), ("Rentang tahun", f"{years[0]}–{years[1]}" if years else DASH),
                 ("Label fraud tersedia", "Ya" if self.d.dataset.get("punya_label") else "Tidak"),
                 ("Baris dengan missing value", fint(sm.get("baris_dengan_missing"))), ("Sel outlier terdeteksi", fint(sm.get("jumlah_sel_outlier"))),
                 ("Status validasi", "Lolos" if val.get("is_valid") else "Tidak lolos")])
        issues = val.get("issues", [])
        if issues:
            self.h2("Temuan validasi")
            order = {"error": 0, "warning": 1, "info": 2}
            rows = [[i["severity"].upper(), i["message"]] for i in sorted(issues, key=lambda i: order.get(i["severity"], 3))[:14]]
            self.table(["Tingkat", "Pesan"], rows, [0.7, 4.0])
            if len(issues) > 14:
                self.small(f"{len(issues) - 14} temuan lain tidak ditampilkan.")

    def eda_section(self) -> None:
        eda = self.d.eda
        if not eda:
            return
        self.h1("Eksplorasi Data")
        dist = eda.get("distribusi_per_tahun")
        if dist:
            self.h2("Proporsi fraud per tahun")
            self.table(["Tahun", "n", "Fraud", "Proporsi", "CI 95% Wilson"], [[r["kelompok"], fint(r["n"]), fint(r["fraud"]), fpct(r["proporsi_fraud"], 2),
                       f"{fpct(r['ci95_bawah'], 1)}–{fpct(r['ci95_atas'], 1)}"] for r in dist], [0.8, 0.9, 0.8, 1.0, 1.6], right=(1, 2, 3))
        g = (eda.get("benford_per_kelompok") or {}).get("semua")
        if g:
            fd, ld = g["first_digit"], g["last_digit"]
            self.h2("Uji Benford")
            self.side_by_side(self.figure(charts.benford_figure(fd, "Digit pertama vs Benford"), 8.2),
                              _rich(f"<b>Digit pertama:</b> MAD = {fnum(fd['mad'], 4)} ({escape(str(fd['conformity']))}); χ² = {fnum(fd['chi2'], 1)}, "
                                    f"db {fd['dof']}, p {fp(fd['p_value'])}. n = {fint(fd['n'])} angka.<br/><br/><b>Digit terakhir:</b> {escape(str(ld['conformity']))} "
                                    f"(χ² = {fnum(ld['chi2'], 1)}, p {fp(ld['p_value'])}).<br/><br/>Pada n besar χ² hampir selalu menolak; MAD (ambang Nigrini "
                                    "0,006 / 0,012 / 0,015) lebih informatif. Penyimpangan bukan bukti fraud, hanya indikasi untuk diperiksa.", self.s.body))
        cmp_rows = eda.get("perbandingan_kelompok")
        if cmp_rows:
            self.h2("Fraud vs non-fraud (Mann-Whitney, koreksi Benjamini-Hochberg)")
            top = sorted([r for r in cmp_rows if r.get("efek_rank_biserial") is not None], key=lambda r: -abs(r["efek_rank_biserial"]))[:8]
            self.table(["Variabel", "Median fraud", "Median non-fraud", "Efek r", "q"], [[r["variabel"], fsmart(r["median_fraud"]), fsmart(r["median_non_fraud"]),
                       fnum(r["efek_rank_biserial"], 3), fp(r["q_value"])] for r in top], [1.6, 1.1, 1.3, 0.8, 0.8], right=(1, 2, 3, 4))

    def prediction_section(self) -> None:
        p = self.d.prediction
        self.h1("Hasil Prediksi")
        self.add_figure(charts.histogram_figure([r["probabilitas"] for r in self.rows], self.thr), 14.5)
        by: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in self.rows:
            by[r["sektor"]].append(r)
        self.h2("Ringkasan per sektor")
        self.table(["Sektor", "n", "Ditandai", "% ditandai", "Rata-rata probabilitas"], [[k, fint(len(v)), fint(sum(x["ditandai"] for x in v)),
                   fpct(sum(x["ditandai"] for x in v) / len(v), 1), fprob(sum(x["probabilitas"] for x in v) / len(v))] for k, v in sorted(by.items())],
                   [1.6, 0.8, 0.9, 1.0, 1.6], right=(1, 2, 3, 4))
        n = max(1, min(int(self.d.top_n), 50))
        top = sorted(self.rows, key=lambda r: -r["probabilitas"])[:n]
        self.h2(f"{len(top)} observasi dengan probabilitas tertinggi")
        has_f = any(r.get("faktor_utama") for r in top)
        head = ["#", "Perusahaan", "Tahun", "Sektor", "Prob.", "Risiko", "Beneish", "Altman Z″"] + (["Faktor utama"] if has_f else [])
        rows = []
        for i, r in enumerate(top, 1):
            row = [str(i), r["company_id"], str(r["tahun"]), r["sektor"], fprob(r["probabilitas"]), r["tingkat_risiko"],
                   "tandai" if r.get("m_flag") else DASH, str(r.get("altman_z2_zona") or DASH)]
            if has_f:
                row.append("; ".join(f"{f['label']} ({'+' if f['kontribusi'] > 0 else '−'})" for f in (r.get("faktor_utama") or [])[:3]) or DASH)
            rows.append(row)
        self.table(head, rows, [0.3, 1.0, 0.6, 1.1, 0.8, 0.8, 0.8, 0.9] + ([2.4] if has_f else []), right=(4,))
        for c in p.get("catatan", []):
            self.small(c)
        self.small("Probabilitas adalah hasil kalibrasi Platt scaling. Beneish M-Score dan Altman Z″ adalah benchmark berbasis teori, bukan bagian model. "
                   "Tingkat risiko: Tinggi bila p ≥ ambang; Sedang bila p ≥ ambang/2; selainnya Rendah.")

    def performance_section(self) -> None:
        ev = self.d.evaluation
        if not ev:
            return
        self.h1("Kinerja Model")
        self.para(f"Evaluasi pada data uji hold-out: {fint(ev.get('n_uji'))} baris dengan {fint(ev.get('fraud_uji'))} fraud. Interval kepercayaan 95% dihitung dengan "
                  f"bootstrap per perusahaan ({fint(ev.get('n_bootstrap'))} replikasi). Dengan jumlah fraud sekecil ini, selisih antar model yang kecil berada dalam derau statistik.")
        boot = ev.get("metrik_uji_bootstrap", [])
        names = [m for m in ("logreg", "logit_mle", "random_forest", "hist_gb", "isolation_forest", "m_score") if any(b["model"] == m for b in boot)]
        get = lambda m, k: next((b for b in boot if b["model"] == m and b["metrik"] == k), None)
        ci = lambda b: f"{fnum(b['estimasi'], 3)} [{fnum(b['ci_bawah'], 3)}; {fnum(b['ci_atas'], 3)}]" if b else DASH
        self.table(["Model", "PR-AUC [CI 95%]", "ROC-AUC [CI 95%]", "Recall@10% [CI 95%]"],
                   [[model_label(m), ci(get(m, "pr_auc")), ci(get(m, "roc_auc")), ci(get(m, "recall_at_10"))] for m in names], [1.6, 1.6, 1.6, 1.6])
        models = [m for m in ("logreg", "random_forest", "hist_gb", "isolation_forest") if m in ev.get("kurva", {})]
        if models:
            self.add_figure(charts.roc_pr_figure(ev["kurva"], models), 16.5, "Kurva ROC dan Precision-Recall pada data uji.")
        cal_key = f"{self.model}:platt" if f"{self.model}:platt" in ev.get("kurva_kalibrasi", {}) else None
        a = self.figure(charts.calibration_figure(ev["kurva_kalibrasi"][cal_key], f"Kalibrasi {self.label} (Platt)"), 8.0) if cal_key else None
        b = self.figure(charts.metric_ci_figure(boot, "pr_auc", [m for m in names if m != "m_score"], "PR-AUC dan CI 95%"), 8.0)
        if a:
            self.side_by_side(a, b)
        else:
            self.add_figure(charts.metric_ci_figure(boot, "pr_auc", [m for m in names if m != "m_score"], "PR-AUC dan CI 95%"), 8.0)
        cal = [r for r in ev.get("kalibrasi", []) if r["model"] == self.model]
        if cal:
            self.h2(f"Kalibrasi {self.label}")
            self.table(["Versi", "Brier", "ECE", "Slope", "CITL"], [[r["versi"], fnum(r["brier"], 4), fnum(r["ece"], 4), fnum(r["slope"], 2), fnum(r["citl"], 2)] for r in cal],
                       [1.2, 1, 1, 1, 1], right=(1, 2, 3, 4))
            self.small("Slope ideal = 1; CITL (calibration-in-the-large) ideal = 0. Kalibrator dilatih pada prediksi out-of-fold data latih.")
        thr = [r for r in ev.get("ambang_keputusan", []) if r["model"] == self.model]
        if thr:
            bi = ev.get("biaya", {})
            self.h2(f"Ambang keputusan dan biaya ({self.label})")
            self.table(["Aturan", "Ambang", "Recall", "Precision", "% ditandai", "Hemat biaya"], [[r["aturan"], fnum(r["threshold"], 3), fpct(r["recall"], 1), fpct(r["precision"], 1),
                       fpct(r["persen_ditandai"], 1), f"{fnum(r['penghematan_biaya_persen'], 0)}%" if r.get("penghematan_biaya_persen") is not None else DASH] for r in thr],
                       [1.5, 0.9, 0.9, 0.9, 1.0, 1.0], right=(1, 2, 3, 4, 5))
            self.small(f"Biaya false negative = {fnum(bi.get('c_fn'), 0)}, false positive = {fnum(bi.get('c_fp'), 0)} (asumsi); ambang Bayes teoretis = {fpct(bi.get('ambang_bayes'), 1)}. "
                       "Ambang dipilih pada data latih (out-of-fold) lalu diterapkan ke data uji.")
        pairs = [r for r in ev.get("perbandingan_model", []) if r["model_a"] in SUPERVISED and r["model_b"] in SUPERVISED]
        if pairs:
            mn = min(r["p_delong_holm"] for r in pairs)
            self.h2("Perbandingan antar model")
            verdict = "tidak menemukan perbedaan signifikan" if mn >= 0.05 else "menemukan perbedaan signifikan"
            tail = ("Karena itu tidak ada dasar statistik untuk menyatakan satu model terbaik; regresi logistik setara dan lebih mudah ditafsirkan."
                    if mn >= 0.05 else "Periksa tabel perbandingan pada aplikasi untuk rinciannya.")
            self.para(f"Uji DeLong (AUC, koreksi Holm) {verdict} antar model terawasi (p Holm terkecil = {fp(mn)}). {tail}")
        d = ev.get("diagnostik_logit")
        if d:
            hl = d["hosmer_lemeshow_latih"]
            bt = d.get("box_tidwell", [])
            self.h2("Diagnostik regresi logistik")
            self.bullets([f"Hosmer-Lemeshow (data latih): χ² = {fnum(hl['statistik'], 2)}, db = {hl['dof']}, p = {fp(hl['p_value'])} → {'model cocok' if hl['p_value'] >= 0.05 else 'model tidak cocok'}.",
                          f"Box-Tidwell: {sum(1 for r in bt if r['linear'])} dari {len(bt)} prediktor memenuhi asumsi linearitas logit (ambang Bonferroni).",
                          f"Cook's Distance maksimum = {fnum(d['pengaruh_ringkasan'].get('cooks_d_maks'), 3)} (tidak ada observasi sangat berpengaruh bila < 1)."])

    def explanation_section(self) -> None:
        ex = self.d.explanation
        if not ex:
            return
        self.h1("Penjelasan Model")
        odds = ex.get("interpretasi_odds_ratio", [])
        if odds:
            self.h2("Odds ratio regresi logistik")
            self.add_figure(charts.forest_figure(odds, "Odds ratio per 1 simpangan baku (CI 95%)"), 16.0)
            self.table(["Fitur", "OR [CI 95%]", "p", "Kesesuaian teori"], [[r["label"], f"{fnum(r['odds_ratio'], 2)} [{fnum(r['ci_bawah'], 2)}; {fnum(r['ci_atas'], 2)}]",
                       fp(r["p_value"]), r["kesesuaian"]] for r in odds], [2.6, 1.7, 0.7, 1.6], right=(1, 2))
            sig = [r for r in odds if r["signifikan"]][:5]
            if sig:
                self.h2("Interpretasi")
                self.bullets([r["interpretasi"] for r in sig])
        key = self.model if self.model in ex.get("shap_global", {}) else ("random_forest" if "random_forest" in ex.get("shap_global", {}) else None)
        if key:
            labels = {r["fitur"]: r["label"] for r in odds}
            rows = sorted(ex["shap_global"][key], key=lambda r: -r["mean_abs_shap"])
            self.h2(f"Kepentingan fitur (SHAP eksak) — {model_label(key)}")
            self.add_figure(charts.hbar_figure([labels.get(r["fitur"], r["fitur"]) for r in rows], [r["porsi"] * 100 for r in rows], "Porsi rata-rata |SHAP| per fitur", "Porsi (%)"), 15.5)
            self.small(f"{ex.get('catatan_sampel', '')} Satuan atribusi: {ex.get('shap_satuan', {}).get(key, DASH)}. Bandingkan peringkat dan porsi antar model, bukan besaran mentah.")

    def limitations_section(self) -> None:
        self.h1("Asumsi, Keterbatasan, dan Bias")
        self.h2("Asumsi")
        self.bullets(["Rasio keuangan dari laporan dapat dipercaya sebagai bahan analisis (garbage in, garbage out).",
                      "Regresi logistik mengasumsikan logit linear terhadap fitur (diuji Box-Tidwell) dan tanpa multikolinearitas berat (VIF).",
                      "Probabilitas dikalibrasi dengan Platt scaling pada prediksi out-of-fold.",
                      "Koefisien Beneish (1999) berasal dari data manufaktur AS dan hanya dipakai sebagai pembanding."])
        self.h2("Keterbatasan dan potensi bias")
        items = ["Jumlah fraud pada data uji kecil, sehingga CI lebar dan perbedaan antar model tidak bermakna.",
                 "Seleksi fitur dilakukan sebelum validasi silang pada data latih, sehingga skor CV sedikit optimistis; gunakan hasil hold-out.",
                 "Label fraud pada data nyata rentan selection bias (hanya fraud yang terdeteksi yang berlabel) dan measurement bias.",
                 "Model tidak menjamin generalisasi ke sektor, periode, atau regulasi yang berbeda; pantau pergeseran data secara berkala.",
                 "Skor risiko adalah indikasi untuk diperiksa lebih lanjut; keputusan audit memerlukan pertimbangan profesional dan bukti tambahan."]
        if self.d.synthetic:
            items.insert(0, "DATA SINTETIS: fitur pembulatan digit dikodekan kuat pada generator; pada data nyata pengaruhnya akan jauh lebih kecil. "
                            "Hasil laporan ini hanya untuk mendemonstrasikan sistem.")
        self.bullets(items)

    def reproducibility_section(self) -> None:
        self.h1("Reproducibility")
        mo, mi = self.d.modeling or {}, self.d.model_info or {}
        meta = mi.get("metadata", {}) if isinstance(mi, dict) else {}
        items = [("Random seed", "42 (seluruh tahap)"), ("Model", self.label)]
        if meta.get("strategi_imbalance"):
            items.append(("Strategi imbalance", str(meta["strategi_imbalance"])))
        if meta.get("parameter"):
            items.append(("Hiperparameter", ", ".join(f"{k}={v}" for k, v in meta["parameter"].items())))
        feats = mo.get("fitur_terpilih") or meta.get("fitur")
        if feats:
            items.append(("Fitur model", ", ".join(feats)))
        if mo.get("versi_library"):
            items.append(("Versi library", " · ".join(f"{k} {v}" for k, v in mo["versi_library"].items())))
        items.append(("Perintah", "python -m scripts.run_feature_engineering && python -m scripts.run_modeling && python -m scripts.run_evaluation"))
        self.kv(items)

    # ------------------------------------------------------------- rakit
    def build(self) -> bytes:
        buf = io.BytesIO()
        doc = BaseDocTemplate(buf, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=2.0 * cm, bottomMargin=1.8 * cm,
                              title=self.d.title, author="Sistem Deteksi Financial Fraud", subject=f"Analisis {self.d.dataset.get('nama', '')}",
                              creator="Sistem Deteksi Financial Fraud (ReportLab)")
        s = self.s

        def cover_page(c: Any, _doc: Any) -> None:
            c.saveState()
            c.setFillColor(NAVY_DARK)
            c.rect(0, 0, PAGE_W, PAGE_H, stroke=0, fill=1)
            c.setFillColor(RED)
            c.rect(0, PAGE_H - 0.9 * cm, PAGE_W, 0.9 * cm, stroke=0, fill=1)
            c.setFillColor(colors.HexColor("#8fa6cc"))
            c.setFont(s.font, 9)
            c.drawString(MARGIN, 1.4 * cm, "Sistem Deteksi Financial Fraud · Skripsi Statistika")
            c.restoreState()

        def body_page(c: Any, d: Any) -> None:
            c.saveState()
            c.setStrokeColor(LINE)
            c.line(MARGIN, PAGE_H - 1.5 * cm, PAGE_W - MARGIN, PAGE_H - 1.5 * cm)
            c.setFont(s.font, 8)
            c.setFillColor(MUTED)
            c.drawString(MARGIN, PAGE_H - 1.25 * cm, self.d.title)
            c.drawRightString(PAGE_W - MARGIN, PAGE_H - 1.25 * cm, str(self.d.dataset.get("nama", "")))
            c.drawString(MARGIN, 1.0 * cm, f"Dibuat {self.d.generated_at}" if self.d.generated_at else "")
            c.drawRightString(PAGE_W - MARGIN, 1.0 * cm, f"Halaman {d.page - 1}")
            c.restoreState()

        frame = Frame(MARGIN, 1.8 * cm, TEXT_W, PAGE_H - 3.8 * cm, id="body", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        cover_frame = Frame(MARGIN, 1.0 * cm, TEXT_W, PAGE_H - 2.0 * cm, id="cover")
        doc.addPageTemplates([PageTemplate(id="cover", frames=[cover_frame], onPage=cover_page), PageTemplate(id="body", frames=[frame], onPage=body_page)])
        self.cover()
        self.summary()
        self.data_section()
        self.eda_section()
        self.prediction_section()
        self.performance_section()
        self.explanation_section()
        self.limitations_section()
        self.reproducibility_section()
        doc.build(self.story)
        return buf.getvalue()


def build_report_pdf(data: ReportData) -> bytes:
    """Buat laporan PDF dari ``ReportData``. Mengembalikan bytes PDF."""
    if not data.prediction.get("hasil"):
        raise ValueError("Laporan membutuhkan hasil prediksi (prediction['hasil']) yang tidak kosong.")
    return _Builder(data).build()
