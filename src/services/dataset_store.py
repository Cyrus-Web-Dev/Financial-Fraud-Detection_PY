"""Penyimpanan dataset unggahan di server, diacu lewat ``dataset_id``.

Frontend multi-langkah (unggah -> EDA -> prediksi) tidak perlu mengirim ulang file besar: cukup membawa id.

Keamanan
--------
* ``dataset_id`` selalu uuid4 hex (32 karakter [0-9a-f]) dan divalidasi dengan regex sebelum menyentuh sistem
  file, sehingga path traversal (``../``) mustahil.
* Hanya data yang SUDAH divalidasi dan dikoersi yang disimpan; jumlah dataset dibatasi (yang terlama dihapus).
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import MAX_STORED_DATASETS
from src.services.errors import ServiceError

ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


class DatasetStore:
    def __init__(self, directory: Path | str, max_items: int = MAX_STORED_DATASETS) -> None:
        self.directory = Path(directory)
        self.max_items = max_items
        self.directory.mkdir(parents=True, exist_ok=True)

    def _paths(self, dataset_id: str) -> tuple[Path, Path]:
        if not isinstance(dataset_id, str) or not ID_PATTERN.match(dataset_id):
            raise ServiceError("DATASET_ID_TIDAK_VALID", "dataset_id tidak valid.", 400)
        return self.directory / f"{dataset_id}.csv", self.directory / f"{dataset_id}.json"

    def save(self, df: pd.DataFrame, name: str, extra: dict | None = None) -> str:
        dataset_id = uuid.uuid4().hex
        data_path, meta_path = self._paths(dataset_id)
        df.to_csv(data_path, index=False)
        meta = {"id": dataset_id, "nama": str(name)[:200], "dibuat": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                "n_baris": int(len(df)), "n_kolom": int(df.shape[1]), **(extra or {})}
        meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        self._evict()
        return dataset_id

    def load(self, dataset_id: str) -> pd.DataFrame:
        data_path, _ = self._paths(dataset_id)
        if not data_path.exists():
            raise ServiceError("DATASET_TIDAK_DITEMUKAN", "Dataset tidak ditemukan atau sudah dihapus. Unggah ulang.", 404)
        return pd.read_csv(data_path)

    def meta(self, dataset_id: str) -> dict:
        _, meta_path = self._paths(dataset_id)
        if not meta_path.exists():
            raise ServiceError("DATASET_TIDAK_DITEMUKAN", "Dataset tidak ditemukan atau sudah dihapus. Unggah ulang.", 404)
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def delete(self, dataset_id: str) -> None:
        for p in self._paths(dataset_id):
            p.unlink(missing_ok=True)

    def list(self) -> list[dict]:
        metas = []
        for p in self.directory.glob("*.json"):
            if ID_PATTERN.match(p.stem):
                metas.append(json.loads(p.read_text(encoding="utf-8")))
        return sorted(metas, key=lambda m: m["dibuat"], reverse=True)

    def _evict(self) -> None:
        metas = self.list()
        for m in metas[self.max_items:]:
            self.delete(m["id"])
