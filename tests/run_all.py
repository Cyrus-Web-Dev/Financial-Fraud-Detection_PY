"""Jalankan semua uji tanpa pytest:  python -m tests.run_all"""
from __future__ import annotations

import importlib
import sys
import traceback

MODULES = ["tests.test_synthetic", "tests.test_benford", "tests.test_eda", "tests.test_validation",
           "tests.test_features", "tests.test_selection", "tests.test_imbalance", "tests.test_logit",
           "tests.test_models", "tests.test_evaluation", "tests.test_diagnostics", "tests.test_explanation",
           "tests.test_services", "tests.test_api_static", "tests.test_api", "tests.test_reporting", "tests.test_pipeline"]


def main() -> int:
    failed, total = 0, 0
    for name in MODULES:
        mod = importlib.import_module(name)
        for fn_name in sorted(n for n in dir(mod) if n.startswith("test_")):
            total += 1
            try:
                getattr(mod, fn_name)()
            except Exception:
                failed += 1
                print(f"GAGAL {name}.{fn_name}")
                traceback.print_exc()
    print(f"\n{total - failed}/{total} uji lulus" + ("" if not failed else f", {failed} gagal"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
