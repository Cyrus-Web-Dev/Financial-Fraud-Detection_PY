"""Riwayat prediksi di SQLite (modul ``sqlite3`` bawaan Python).

Semua query memakai parameter terikat (bukan interpolasi string) sehingga aman dari SQL injection.
Koneksi dibuka per operasi: aman dipakai dari thread pool FastAPI.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    dibuat        TEXT    NOT NULL,
    model         TEXT    NOT NULL,
    sumber        TEXT    NOT NULL,
    company_id    TEXT,
    tahun         INTEGER,
    probabilitas  REAL    NOT NULL,
    ambang        REAL    NOT NULL,
    ditandai      INTEGER NOT NULL,
    tingkat_risiko TEXT   NOT NULL,
    rincian       TEXT
);
CREATE INDEX IF NOT EXISTS idx_pred_dibuat  ON predictions(dibuat);
CREATE INDEX IF NOT EXISTS idx_pred_company ON predictions(company_id);
"""
COLUMNS = ("id", "dibuat", "model", "sumber", "company_id", "tahun", "probabilitas", "ambang", "ditandai",
           "tingkat_risiko", "rincian")


class HistoryRepository:
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, timeout=10)
        con.row_factory = sqlite3.Row
        return con

    def add_many(self, records: list[dict[str, Any]], model: str, source: str, threshold: float) -> int:
        """Simpan hasil prediksi. Tiap record butuh: company_id, tahun, probabilitas, ditandai, tingkat_risiko."""
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        rows = [(now, model, source, None if r.get("company_id") is None else str(r["company_id"])[:100],
                 None if r.get("tahun") is None else int(r["tahun"]), float(r["probabilitas"]), float(threshold),
                 int(bool(r["ditandai"])), str(r["tingkat_risiko"]),
                 json.dumps(r.get("rincian"), ensure_ascii=False) if r.get("rincian") is not None else None)
                for r in records]
        with closing(self._connect()) as con, con:
            con.executemany("INSERT INTO predictions (dibuat, model, sumber, company_id, tahun, probabilitas, ambang, "
                            "ditandai, tingkat_risiko, rincian) VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
        return len(rows)

    @staticmethod
    def _row(r: sqlite3.Row) -> dict[str, Any]:
        d = {c: r[c] for c in COLUMNS}
        d["ditandai"] = bool(d["ditandai"])
        d["rincian"] = json.loads(d["rincian"]) if d["rincian"] else None
        return d

    def list(self, limit: int = 50, offset: int = 0, model: str | None = None, company_id: str | None = None,
             only_flagged: bool = False) -> list[dict[str, Any]]:
        limit, offset = max(1, min(int(limit), 500)), max(0, int(offset))
        where, params = [], []
        if model:
            where.append("model = ?"); params.append(model)
        if company_id:
            where.append("company_id = ?"); params.append(company_id)
        if only_flagged:
            where.append("ditandai = 1")
        sql = "SELECT * FROM predictions" + (" WHERE " + " AND ".join(where) if where else "") + \
              " ORDER BY id DESC LIMIT ? OFFSET ?"
        with closing(self._connect()) as con:
            return [self._row(r) for r in con.execute(sql, [*params, limit, offset]).fetchall()]

    def get(self, record_id: int) -> dict[str, Any] | None:
        with closing(self._connect()) as con:
            r = con.execute("SELECT * FROM predictions WHERE id = ?", (int(record_id),)).fetchone()
        return self._row(r) if r else None

    def count(self) -> int:
        with closing(self._connect()) as con:
            return int(con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])

    def clear(self) -> int:
        with closing(self._connect()) as con, con:
            n = con.execute("DELETE FROM predictions").rowcount
        return int(n)
