"""Uji orkestrasi pipeline dengan tahap palsu. Jalankan: python -m tests.test_pipeline"""
from __future__ import annotations

import io
import contextlib

from scripts.run_pipeline import Stage, default_stages, format_summary, main, run_pipeline, select_stages


def _stages(trace: list[str], fail: str | None = None, exit_on: str | None = None) -> list[Stage]:
    def make(k: str):
        def run() -> None:
            trace.append(k)
            if k == fail:
                raise RuntimeError("meledak")
            if k == exit_on:
                raise SystemExit("prasyarat hilang")
        return run
    return [Stage(k, f"Tahap {k}", make(k)) for k in ("a", "b", "c", "d")]


def test_runs_in_order_with_timing_and_summary() -> None:
    trace: list[str] = []
    ticks = iter(range(0, 100, 2))
    res = run_pipeline(_stages(trace), log=lambda _m: None, clock=lambda: float(next(ticks)))
    assert trace == ["a", "b", "c", "d"] and all(r.ok for r in res) and [r.seconds for r in res] == [2.0] * 4
    s = format_summary(res, 4)
    assert "SEMUA TAHAP BERHASIL" in s and "Total 8.0 dtk" in s and s.count("OK") == 4


def test_stops_on_first_failure_unless_keep_going() -> None:
    trace: list[str] = []
    res = run_pipeline(_stages(trace, fail="b"), log=lambda _m: None)
    assert trace == ["a", "b"] and [r.ok for r in res] == [True, False] and "RuntimeError: meledak" in (res[1].error or "")
    s = format_summary(res, 4)
    assert "GAGAL" in s and "2 tahap tidak dijalankan" in s and "ADA TAHAP YANG GAGAL" in s
    trace2: list[str] = []
    res2 = run_pipeline(_stages(trace2, fail="b"), keep_going=True, log=lambda _m: None)
    assert trace2 == ["a", "b", "c", "d"] and [r.ok for r in res2] == [True, False, True, True]


def test_system_exit_from_a_stage_is_reported_not_propagated() -> None:
    res = run_pipeline(_stages([], exit_on="a"), log=lambda _m: None)
    assert len(res) == 1 and not res[0].ok and "prasyarat hilang" in (res[0].error or "")


def test_keyboard_interrupt_is_not_swallowed() -> None:
    def boom() -> None:
        raise KeyboardInterrupt
    try:
        run_pipeline([Stage("x", "x", boom)], log=lambda _m: None)
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("Ctrl+C harus diteruskan")


def test_select_only_and_skip_and_validation() -> None:
    st = _stages([])
    assert [s.key for s in select_stages(st, ["b", "d"])] == ["b", "d"]
    assert [s.key for s in select_stages(st, None, ["a", "c"])] == ["b", "d"]
    assert [s.key for s in select_stages(st, ["a", "b"], ["b"])] == ["a"]
    for args in ((["zzz"], ()), (None, ["zzz"]), (["a"], ["a"])):
        try:
            select_stages(st, *args)
        except ValueError:
            continue
        raise AssertionError(args)


def test_default_stages_are_complete_and_importable() -> None:
    keys = [s.key for s in default_stages()]
    assert keys == ["data", "eda", "fitur", "model", "evaluasi", "penjelasan", "fixtures", "laporan"]
    import importlib
    for mod in ("generate_sample_data", "run_eda", "run_feature_engineering", "run_modeling", "run_evaluation", "run_explanation",
                "export_frontend_fixtures", "make_report"):
        assert callable(importlib.import_module(f"scripts.{mod}").main), mod


def test_cli_list_and_error_codes() -> None:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert main(["--list"]) == 0
        assert main(["--only", "tidak_ada"]) == 2
    assert "evaluasi" in out.getvalue() and "Kesalahan" in out.getvalue()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"OK  {t.__name__}")
    print(f"\n{len(tests)} uji lulus.")
