"""Kesalahan terstruktur untuk lapisan layanan (dipetakan ke respons HTTP oleh lapisan API)."""
from __future__ import annotations

from typing import Any


class ServiceError(Exception):
    """Kesalahan yang aman ditampilkan ke pengguna (pesan Bahasa Indonesia, tanpa detail internal)."""

    def __init__(self, code: str, message: str, status: int = 400, details: Any = None) -> None:
        super().__init__(message)
        self.code, self.message, self.status, self.details = code, message, status, details

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kode": self.code, "pesan": self.message}
        if self.details is not None:
            out["rincian"] = self.details
        return out
