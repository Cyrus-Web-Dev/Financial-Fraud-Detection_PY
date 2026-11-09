"""Pemeriksaan STATIS lapisan FastAPI (tanpa memasang fastapi): sintaks, kontrak ke layanan, dan skema permintaan.

Tes ini membaca ``src/api/main.py`` sebagai pohon sintaks (AST) dan memastikan:
* setiap endpoint hanya memanggil method yang BENAR-BENAR ada pada ``FraudDetectionService``,
* jumlah dan nama argumen pemanggilan cocok dengan tanda tangan method (tidak ada typo atau argumen kurang/lebih),
* atribut yang dibaca dari objek permintaan ada pada skema Pydantic yang sesuai,
* tidak ada rute ganda dan semua rute ada di bawah /api.
Jalankan: python -m tests.test_api_static
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

from src.services.facade import FraudDetectionService

MAIN = Path(__file__).resolve().parent.parent / "src" / "api" / "main.py"
HTTP = {"get", "post", "delete", "put", "patch"}


def _tree() -> ast.Module:
    return ast.parse(MAIN.read_text(encoding="utf-8"), filename=str(MAIN))


def _routes(tree: ast.Module) -> list[tuple[str, str, ast.FunctionDef]]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute) and dec.func.attr in HTTP \
                        and isinstance(dec.func.value, ast.Name) and dec.func.value.id == "app":
                    out.append((dec.func.attr.upper(), ast.literal_eval(dec.args[0]), node))
    return out


def _schemas(tree: ast.Module) -> dict[str, set[str]]:
    schemas = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            schemas[node.name] = {n.target.id for n in node.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}
    return schemas


def test_compiles_and_routes_are_well_formed() -> None:
    compile(MAIN.read_text(encoding="utf-8"), str(MAIN), "exec")
    routes = _routes(_tree())
    assert len(routes) >= 15
    keys = [(m, p) for m, p, _ in routes]
    assert len(keys) == len(set(keys)), "rute ganda"
    assert all(p.startswith("/api/") for _, p, _ in routes)
    for _, path, fn in routes:                                   # parameter path harus ada pada argumen fungsi
        import re
        for name in re.findall(r"{(\w+)}", path):
            assert name in {a.arg for a in fn.args.args}, (path, name)


def test_every_endpoint_calls_existing_service_method_with_matching_arguments() -> None:
    tree = _tree()
    schemas = _schemas(tree)
    for method, path, fn in _routes(tree):
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and isinstance(n.func.value, ast.Name) and n.func.value.id == "svc"]
        assert calls, f"{method} {path} tidak memanggil layanan"
        assert len({c.func.attr for c in calls}) == 1, f"{method} {path} memanggil lebih dari satu method layanan"
        call = calls[0]
        target = getattr(FraudDetectionService, call.func.attr, None)
        assert callable(target), f"{method} {path}: svc.{call.func.attr} tidak ada"
        params = [p for p in inspect.signature(target).parameters.values() if p.name != "self"]
        required = [p for p in params if p.default is inspect.Parameter.empty]
        n_pos = len(call.args)
        kw = {k.arg for k in call.keywords}
        assert kw <= {p.name for p in params}, f"{path}: argumen kunci tidak dikenal {kw - {p.name for p in params}}"
        assert n_pos <= len(params), f"{path}: terlalu banyak argumen posisi"
        for i, a in enumerate(call.args):                       # nama yang diberikan harus sama dengan nama parameter pada posisi itu
            given = a.id if isinstance(a, ast.Name) else a.attr if isinstance(a, ast.Attribute) else None
            if given is not None:
                assert given == params[i].name, f"{path}: argumen posisi ke-{i + 1} '{given}' != parameter '{params[i].name}'"
        supplied = {p.name for p in params[:n_pos]} | kw
        assert {p.name for p in required} <= supplied, f"{path}: argumen wajib hilang {({p.name for p in required} - supplied)}"
        # atribut dari objek permintaan harus ada pada skema
        for arg in fn.args.args:
            if arg.arg == "req" and isinstance(arg.annotation, ast.Name):
                fields = schemas[arg.annotation.id]
                used = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "req"}
                assert used <= fields, f"{path}: req.{used - fields} tidak ada pada {arg.annotation.id}"


def test_request_schemas_align_with_service_parameters() -> None:
    schemas = _schemas(_tree())
    sig = lambda name: set(inspect.signature(getattr(FraudDetectionService, name)).parameters) - {"self"}
    assert schemas["PredictDatasetRequest"] <= sig("predict_dataset")
    assert schemas["PredictRowsRequest"] <= sig("predict_rows")
    assert schemas["WhatIfRequest"] <= sig("what_if")
    assert schemas["ReportRequest"] <= sig("generate_report")
    # nilai default model pada skema harus model yang dikenal
    from src.services.registry import PREDICT_MODELS
    src = MAIN.read_text(encoding="utf-8")
    default = [l for l in src.splitlines() if l.startswith("DEFAULT_MODEL")][0].split("=")[1].strip().strip('"')
    assert default in PREDICT_MODELS


def test_main_has_no_business_logic_imports() -> None:
    """Lapisan HTTP tidak boleh mengimpor modul analisis (sains data) secara langsung."""
    forbidden = ("pandas", "numpy", "sklearn", "scipy", "src.models", "src.features", "src.evaluation", "src.explanation")
    for node in ast.walk(_tree()):
        names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module or ""] if isinstance(node, ast.ImportFrom) else []
        for n in names:
            assert not any(n == f or n.startswith(f + ".") for f in forbidden), f"main.py mengimpor {n}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"OK  {t.__name__}")
    print(f"\n{len(tests)} uji lulus.")
