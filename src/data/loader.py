"""Pemuat data CSV / Excel dengan pemeriksaan keamanan dasar.

* Batas ukuran file (``MAX_UPLOAD_MB``)
* Deteksi otomatis pemisah CSV (koma, titik koma, tab) dan encoding
* Normalisasi nama kolom: huruf kecil, spasi/strip -> underscore
"""
from __future__ import annotations

import io
import re
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from src.config import MAX_UPLOAD_MB

ALLOWED_EXTENSIONS: tuple[str, ...] = (".csv", ".xlsx")


class DataLoadError(ValueError):
    """Kesalahan saat membaca file data (pesan dalam Bahasa Indonesia)."""


def normalize_column_name(name: object) -> str:
    """'Total Assets ' -> 'total_assets'; 'Total-Assets' -> 'total_assets'."""
    text = str(name).strip().lower()
    text = re.sub(r"[\s\-]+", "_", text)
    return text


def _detect_delimiter(text_head: str) -> str:
    header = text_head.splitlines()[0] if text_head else ""
    counts = {d: header.count(d) for d in (",", ";", "\t")}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def _decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise DataLoadError("Encoding file tidak dikenali. Simpan ulang sebagai CSV UTF-8.")


def load_dataset(
    source: str | Path | bytes | BinaryIO,
    filename: str | None = None,
    max_mb: int = MAX_UPLOAD_MB,
) -> pd.DataFrame:
    """Baca CSV/XLSX dari path, bytes, atau file-like object.

    Parameters
    ----------
    source : path file, isi file (bytes), atau objek file biner.
    filename : nama file asli (wajib bila ``source`` berupa bytes/file-like)
        untuk menentukan format dari ekstensi.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise DataLoadError(f"File tidak ditemukan: {path.name}")
        filename = filename or path.name
        raw = path.read_bytes()
    elif isinstance(source, (bytes, bytearray)):
        raw = bytes(source)
    else:
        raw = source.read()

    if not filename:
        raise DataLoadError("Nama file diperlukan untuk menentukan format (.csv atau .xlsx).")
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise DataLoadError(
            f"Format '{ext or '(tanpa ekstensi)'}' tidak didukung. Gunakan .csv atau .xlsx."
        )
    if len(raw) == 0:
        raise DataLoadError("File kosong.")
    if len(raw) > max_mb * 1024 * 1024:
        raise DataLoadError(f"Ukuran file melebihi batas {max_mb} MB.")

    try:
        if ext == ".csv":
            text = _decode(raw)
            sep = _detect_delimiter(text[:4096])
            header = pd.read_csv(io.StringIO(text), sep=sep, header=None, nrows=1, dtype=str)
            df = pd.read_csv(io.StringIO(text), sep=sep)
        else:
            header = pd.read_excel(io.BytesIO(raw), engine="openpyxl", header=None, nrows=1, dtype=str)
            df = pd.read_excel(io.BytesIO(raw), engine="openpyxl")
    except DataLoadError:
        raise
    except Exception as exc:  # pandas/openpyxl melempar beragam tipe error
        raise DataLoadError(f"File tidak dapat dibaca: {exc}") from exc

    if df.shape[0] == 0 or df.shape[1] == 0:
        raise DataLoadError("File tidak berisi data.")

    # Periksa duplikasi pada header MENTAH: pandas otomatis mengganti 'a,a' menjadi 'a','a.1'
    # sehingga duplikasi tidak akan terlihat setelah file dibaca.
    raw_names = pd.Series([normalize_column_name(c) for c in header.iloc[0].tolist()])
    if raw_names.duplicated().any():
        dup = sorted(set(raw_names[raw_names.duplicated()]))
        raise DataLoadError(f"Nama kolom ganda: {', '.join(dup)}")

    df.columns = [normalize_column_name(c) for c in df.columns]
    return df
