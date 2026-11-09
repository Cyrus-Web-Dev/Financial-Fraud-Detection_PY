"""Uji laporan PDF (ReportLab + matplotlib). Jalankan: python -m tests.test_reporting"""
from __future__ import annotations

import copy
import io
import json
import tempfile
from pathlib import Path

from src.config import MODELS_DIR, PROCESSED_DIR, SAMPLE_CLEAN_FILE
from src.reporting import charts
from src.reporting.builder import ReportData, build_report_pdf
from src.reporting.fmt import fint, fnum, fp, fpct, fprob, fsmart, model_label
from src.services.dataset_store import DatasetStore
from src.services.errors import ServiceError
from src.services.facade import FraudDetectionService
from src.services.history import HistoryRepository

try:
    import pypdf
    HAVE_PDF = True
except ImportError:          # pypdf hanya dipakai untuk memeriksa hasil
    HAVE_PDF = False

FIX = Path(__file__).resolve().parent.parent / "frontend" / "tests" / "fixtures"
NEEDED = [PROCESSED_DIR / n for n in ("eda_summary.json", "evaluation_results.json", "explanation_results.json", "modeling_results.json")] \
    + [FIX / "upload_sample.json", FIX / "predict_dataset.json", MODELS_DIR / "random_forest.json"]


def _ready() -> bool:
    return all(p.exists() for p in NEEDED)


def _j(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def _data(**over) -> ReportData:
    up, pred = _j(FIX / "upload_sample.json"), _j(FIX / "predict_dataset.json")
    kw = dict(dataset={"nama": up["nama"], "n_baris": up["n_baris"], "punya_label": up["punya_label"], "validasi": up["validasi"]}, prediction=pred,
              eda=_j(PROCESSED_DIR / "eda_summary.json"), evaluation=_j(PROCESSED_DIR / "evaluation_results.json"),
              explanation=_j(PROCESSED_DIR / "explanation_results.json"), modeling=_j(PROCESSED_DIR / "modeling_results.json"),
              model_info={"metadata": _j(MODELS_DIR / "random_forest.json")}, top_n=20, synthetic=True, generated_at="2026-01-02 03:04")
    kw.update(over)
    return ReportData(**kw)


def _text(pdf: bytes) -> list[str]:
    return [(pg.extract_text() or "") for pg in pypdf.PdfReader(io.BytesIO(pdf)).pages]


def _raises(code: str, fn) -> ServiceError:
    try:
        fn()
    except ServiceError as e:
        assert e.code == code, (e.code, e.message)
        return e
    raise AssertionError(f"ServiceError({code}) tidak muncul")


# ------------------------------ pemformat ---------------------------------
def test_formatters_indonesian_style() -> None:
    assert fnum(1234567.891, 2) == "1.234.567,89" and fnum(0.5, 3) == "0,500" and fnum(-1234.5, 1) == "-1.234,5"
    assert fint(1234.6) == "1.235" and fpct(0.0612) == "6,1%" and fpct(0.5, 0) == "50%"
    assert fprob(0.0004) == "< 0,1%" and fprob(0.005) == "0,50%" and fprob(0.1234) == "12,3%" and fprob(0) == "0,00%"
    assert fp(0.0004) == "< 0,001" and fp(0.0527) == "0,053"
    assert fsmart(12345.6) == "12.346" and fsmart(1.2345) == "1,23" and fsmart(0) == "0" and fsmart(0.00012) == "1,2e-04"
    for f in (fnum, fint, fpct, fprob, fp, fsmart):
        assert all(f(v) == "–" for v in (None, float("nan"), float("inf"), "x", True)), f.__name__
    assert model_label("random_forest") == "Random Forest" and model_label("xyz") == "xyz"


def test_charts_return_valid_png_with_dimensions() -> None:
    if not _ready():
        print("      (dilewati: artefak belum ada)"); return
    ev, ex, eda = _j(PROCESSED_DIR / "evaluation_results.json"), _j(PROCESSED_DIR / "explanation_results.json"), _j(PROCESSED_DIR / "eda_summary.json")
    figs = [charts.roc_pr_figure(ev["kurva"], ["logreg", "random_forest"]), charts.calibration_figure(ev["kurva_kalibrasi"]["random_forest:platt"], "x"),
            charts.metric_ci_figure(ev["metrik_uji_bootstrap"], "pr_auc", ["logreg", "random_forest", "tidak_ada"], "x"),
            charts.hbar_figure(["a", "b" * 60], [1.0, -2.0], "x", "y", signed=True), charts.forest_figure(ex["interpretasi_odds_ratio"], "x"),
            charts.histogram_figure([0.001, 0.5, 1.0, 0.0], 0.09), charts.benford_figure(eda["benford_per_kelompok"]["semua"]["first_digit"], "x")]
    for png, w, h in figs:
        assert png[:8] == b"\x89PNG\r\n\x1a\n" and w > 200 and h > 100 and len(png) > 5000


# ----------------------------- isi PDF -------------------------------------
def test_full_report_structure_and_content() -> None:
    if not (_ready() and HAVE_PDF):
        print("      (dilewati: artefak atau pypdf belum ada)"); return
    pdf = build_report_pdf(_data())
    assert pdf[:5] == b"%PDF-"
    reader = pypdf.PdfReader(io.BytesIO(pdf))
    pages = _text(pdf)
    assert 7 <= len(pages) <= 14 and reader.metadata.title == "Laporan Analisis Indikasi Fraud Laporan Keuangan"
    alltext = "\n".join(pages)
    for phrase in ("Ringkasan Eksekutif", "Data dan Validasi", "Eksplorasi Data", "Hasil Prediksi", "Kinerja Model", "Penjelasan Model",
                   "Asumsi, Keterbatasan, dan Bias", "Reproducibility", "SINTETIS", "Uji Benford", "Odds ratio", "Hosmer-Lemeshow"):
        assert phrase in alltext, phrase
    assert "3.600 observasi" in alltext and "600 perusahaan" in alltext and "2018–2023" in alltext          # angka bergaya Indonesia
    assert "Faktor utama" not in alltext                                                                      # fixture tanpa faktor utama
    import re
    nums = [int(x) for x in re.findall(r"^(\d)\. [A-Z]", alltext, flags=re.M)]
    assert nums == sorted(nums) and nums[:8] == list(range(1, 9)), nums                                       # penomoran berurutan
    fonts = {str(v.get_object().get("/BaseFont", "")) for pg in reader.pages for v in (pg["/Resources"].get("/Font") or {}).values()}
    assert any("DejaVu" in f for f in fonts), fonts                                                           # font Unicode tertanam
    assert sum(len(pg.images) for pg in reader.pages) >= 7                                                    # grafik tertanam
    assert "χ²" in alltext and "≥" in alltext                                                                 # glyph matematis tampil


def test_deterministic_text() -> None:
    if not (_ready() and HAVE_PDF):
        print("      (dilewati)"); return
    assert _text(build_report_pdf(_data())) == _text(build_report_pdf(_data()))


def test_optional_sections_are_skipped_and_numbering_stays_contiguous() -> None:
    if not (_ready() and HAVE_PDF):
        print("      (dilewati)"); return
    pdf = build_report_pdf(_data(eda=None, evaluation=None, explanation=None, modeling=None, model_info=None))
    txt = "\n".join(_text(pdf))
    assert "Kinerja Model" not in txt and "Penjelasan Model" not in txt and "Eksplorasi Data" not in txt
    import re
    assert [int(x) for x in re.findall(r"^(\d)\. [A-Z]", txt, flags=re.M)] == [1, 2, 3, 4, 5]               # nomor tidak melompat
    assert "Reproducibility" in txt and "Random seed" in txt


def test_unlabeled_dataset_and_unsafe_text_and_scope() -> None:
    if not (_ready() and HAVE_PDF):
        print("      (dilewati)"); return
    d = _data(); d.dataset["punya_label"] = False
    d.eda = {k: v for k, v in d.eda.items() if k not in ("distribusi_kelas", "distribusi_per_tahun", "distribusi_per_sektor", "perbandingan_kelompok")}
    d.prediction = copy.deepcopy(d.prediction)
    d.prediction["hasil"][0]["company_id"] = "A&B <PT> \"x\""                                                  # karakter khusus XML
    d.prediction["hasil"][0]["probabilitas"] = 0.9999
    d.scope = "tahun 2023"
    txt = "\n".join(_text(build_report_pdf(d)))
    assert "A&B <PT>" in txt and "Catatan cakupan" in txt and "dari 3.600 observasi (tahun 2023)" in txt
    assert "berlabel fraud" not in txt


def test_edge_cases_no_flagged_and_top_n_clamp_and_empty() -> None:
    if not (_ready() and HAVE_PDF):
        print("      (dilewati)"); return
    d = _data(top_n=100000)
    txt = "\n".join(_text(build_report_pdf(d)))
    assert "50 observasi dengan probabilitas tertinggi" in txt                                              # dibatasi 50
    d2 = _data(top_n=0)
    assert "1 observasi dengan probabilitas tertinggi" in "\n".join(_text(build_report_pdf(d2)))
    d3 = _data(); d3.prediction = copy.deepcopy(d3.prediction)
    for r in d3.prediction["hasil"]:
        r.update(ditandai=False, tingkat_risiko="Rendah", probabilitas=0.001)
    d3.prediction["n_ditandai"] = 0
    assert "menandai 0 dari" in "\n".join(_text(build_report_pdf(d3)))
    d4 = _data(); d4.prediction = {**d4.prediction, "hasil": []}
    try:
        build_report_pdf(d4)
    except ValueError:
        pass
    else:
        raise AssertionError("hasil kosong harus ditolak")


# ------------------------------ layanan -------------------------------------
def test_facade_generate_report_end_to_end() -> None:
    if not (_ready() and HAVE_PDF and (PROCESSED_DIR / "features.csv").exists() and SAMPLE_CLEAN_FILE.exists()):
        print("      (dilewati: artefak belum ada)"); return
    with tempfile.TemporaryDirectory() as d:
        s = FraudDetectionService(store=DatasetStore(Path(d) / "u"), history=HistoryRepository(Path(d) / "h.db"))
        did = s.load_sample()["dataset_id"]
        assert s.store.meta(did)["sintetis"] is True
        pdf, name = s.generate_report(did, "logit_mle", 0.2, 2023, 5)
        assert pdf[:5] == b"%PDF-" and name.startswith("laporan_fraud_logit_mle_") and name.endswith(".pdf")
        txt = "\n".join(_text(pdf))
        assert "SINTETIS" in txt and "tahun 2023" in txt and "Faktor utama" in txt and "20,0%" in txt
        assert "5 observasi dengan probabilitas tertinggi" in txt
        assert s.history.count() == 0                                                                        # laporan tidak mengubah riwayat
        for bad_n in (0, 51, "5", True, 2.5):
            _raises("TOP_N_TIDAK_VALID", lambda b=bad_n: s.generate_report(did, "logreg", None, None, b))
        _raises("TAHUN_TIDAK_ADA", lambda: s.generate_report(did, "logreg", None, 1999, 5))
        _raises("TAHUN_TIDAK_VALID", lambda: s.generate_report(did, "logreg", None, "2023", 5))
        _raises("MODEL_TIDAK_DIKENAL", lambda: s.generate_report(did, "isolation_forest", None, None, 5))
        _raises("AMBANG_TIDAK_VALID", lambda: s.generate_report(did, "logreg", 2.0, None, 5))
        _raises("DATASET_ID_TIDAK_VALID", lambda: s.generate_report("../x", "logreg", None, None, 5))
        _raises("DATASET_TIDAK_DITEMUKAN", lambda: s.generate_report("a" * 32, "logreg", None, None, 5))


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"OK  {t.__name__}")
    print(f"\n{len(tests)} uji lulus.")
