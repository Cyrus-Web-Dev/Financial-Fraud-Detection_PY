"""Jalankan seluruh pipeline (Tahap 1-9) dalam satu perintah.

    python -m scripts.run_pipeline              # lengkap (sekitar 8-10 menit)
    python -m scripts.run_pipeline --quick      # pemodelan cepat (nested CV 3 lipatan, tanpa ablasi)
    python -m scripts.run_pipeline --list       # daftar tahap
    python -m scripts.run_pipeline --only evaluasi,laporan
    python -m scripts.run_pipeline --skip fixtures

Tahap berhenti pada kegagalan pertama (kecuali --keep-going) dan ringkasan waktu dicetak di akhir.
"""
from __future__ import annotations

import argparse
import time
import traceback
from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass
class Stage:
    key: str
    title: str
    run: Callable[[], object]


@dataclass
class StageResult:
    key: str
    title: str
    ok: bool
    seconds: float
    error: str | None = None


def default_stages(quick: bool = False) -> list[Stage]:
    """Daftar tahap bawaan; impor dilakukan malas agar ``--list`` tetap cepat."""
    def data() -> None:
        from scripts import generate_sample_data
        generate_sample_data.main()

    def eda() -> None:
        from scripts import run_eda
        run_eda.main()

    def fitur() -> None:
        from scripts import run_feature_engineering
        run_feature_engineering.main()

    def model() -> None:
        from scripts import run_modeling
        run_modeling.main(quick=quick)

    def evaluasi() -> None:
        from scripts import run_evaluation
        run_evaluation.main()

    def penjelasan() -> None:
        from scripts import run_explanation
        run_explanation.main()

    def fixtures() -> None:
        from scripts import export_frontend_fixtures
        export_frontend_fixtures.main()

    def laporan() -> None:
        from scripts import make_report
        make_report.main()

    return [Stage("data", "Data sampel sintetis", data), Stage("eda", "Validasi dan EDA", eda),
            Stage("fitur", "Feature engineering dan seleksi fitur", fitur), Stage("model", "Pemodelan", model),
            Stage("evaluasi", "Evaluasi dan diagnostik", evaluasi), Stage("penjelasan", "Explainability", penjelasan),
            Stage("fixtures", "Fixture frontend", fixtures), Stage("laporan", "Laporan PDF contoh", laporan)]


def select_stages(stages: Sequence[Stage], only: Sequence[str] | None = None, skip: Sequence[str] = ()) -> list[Stage]:
    keys = [s.key for s in stages]
    for name in list(only or []) + list(skip):
        if name not in keys:
            raise ValueError(f"Tahap '{name}' tidak dikenal. Pilihan: {', '.join(keys)}")
    chosen = [s for s in stages if (not only or s.key in only) and s.key not in skip]
    if not chosen:
        raise ValueError("Tidak ada tahap yang dipilih.")
    return chosen


def run_pipeline(stages: Sequence[Stage], keep_going: bool = False, log: Callable[[str], None] = print,
                 clock: Callable[[], float] = time.perf_counter) -> list[StageResult]:
    """Jalankan tahap berurutan. Mengembalikan hasil per tahap; berhenti pada kegagalan pertama kecuali ``keep_going``."""
    results: list[StageResult] = []
    for i, st in enumerate(stages, 1):
        log(f"\n=== [{i}/{len(stages)}] {st.title} ({st.key}) ===")
        t0 = clock()
        try:
            st.run()
            results.append(StageResult(st.key, st.title, True, clock() - t0))
        except BaseException as exc:  # noqa: BLE001 - SystemExit dari skrip (mis. prasyarat hilang) juga harus dilaporkan
            if isinstance(exc, KeyboardInterrupt):
                raise
            msg = f"{type(exc).__name__}: {exc}"
            log(traceback.format_exc() if not isinstance(exc, SystemExit) else f"Berhenti: {exc}")
            results.append(StageResult(st.key, st.title, False, clock() - t0, msg))
            if not keep_going:
                break
    return results


def format_summary(results: Sequence[StageResult], total_planned: int) -> str:
    lines = ["\nRINGKASAN PIPELINE"]
    for r in results:
        lines.append(f"  {'OK    ' if r.ok else 'GAGAL '} {r.key:11s} {r.seconds:7.1f} dtk  {r.title}" + (f"  -> {r.error}" if r.error else ""))
    skipped = total_planned - len(results)
    if skipped:
        lines.append(f"  ({skipped} tahap tidak dijalankan karena tahap sebelumnya gagal)")
    ok = all(r.ok for r in results) and skipped == 0
    lines.append(f"  Total {sum(r.seconds for r in results):.1f} dtk · {'SEMUA TAHAP BERHASIL' if ok else 'ADA TAHAP YANG GAGAL'}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="pemodelan cepat")
    ap.add_argument("--list", action="store_true", help="tampilkan daftar tahap")
    ap.add_argument("--only", default="", help="jalankan hanya tahap ini (dipisah koma)")
    ap.add_argument("--skip", default="", help="lewati tahap ini (dipisah koma)")
    ap.add_argument("--keep-going", action="store_true", help="lanjut walau ada tahap yang gagal")
    a = ap.parse_args(argv)
    stages = default_stages(quick=a.quick)
    if a.list:
        for s in stages:
            print(f"{s.key:11s} {s.title}")
        return 0
    split = lambda v: [x.strip() for x in v.split(",") if x.strip()]
    try:
        chosen = select_stages(stages, split(a.only) or None, split(a.skip))
    except ValueError as exc:
        print(f"Kesalahan: {exc}")
        return 2
    results = run_pipeline(chosen, keep_going=a.keep_going)
    print(format_summary(results, len(chosen)))
    return 0 if all(r.ok for r in results) and len(results) == len(chosen) else 1


if __name__ == "__main__":
    raise SystemExit(main())
