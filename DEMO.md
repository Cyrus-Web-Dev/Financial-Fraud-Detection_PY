# Panduan Demo 5–7 Menit

Tujuan: menunjukkan kepada dosen bahwa sistem ini **metodologis**, **dapat dijelaskan**, dan **jujur tentang keterbatasannya**.

## Persiapan (sebelum demo)
```bash
pip install -r requirements.txt
python -m scripts.run_pipeline --quick        # ±6 menit; hasilkan model, evaluasi, penjelasan
uvicorn src.api.main:app --port 8000          # terminal 1
cd frontend && npm install && npm run dev     # terminal 2 → http://localhost:5173
```
Cadangan bila jaringan/laptop bermasalah: `python -m scripts.make_report` menghasilkan `reports/*.pdf`; `reports/contoh_laporan.pdf` sudah disertakan.

## Alur (≈ 6,5 menit)
| Menit | Halaman | Yang dilakukan dan dikatakan |
|---|---|---|
| 0:00 | **Beranda** | "Sistem mendeteksi *indikasi* fraud laporan keuangan; bukan pembuktian." Tunjuk catatan: data sampel **sintetis**. Klik **Coba dengan data sampel**. |
| 0:40 | **Unggah & Validasi** | Muat juga sampel "kotor": sistem melaporkan missing value dan outlier tanpa menolak data. Sebut pemeriksaan identitas neraca dan kontinuitas panel. |
| 1:30 | **Eksplorasi Data** | Tab *Distribusi kelas*: fraud ±6% → accuracy menyesatkan. Tab *Uji Benford*: digit pertama sesuai Benford (MAD), tetapi **digit terakhir** tidak seragam pada fraud. Jelaskan: χ² sensitif pada n besar, MAD lebih informatif. |
| 2:30 | **Feature Engineering** | Nilai mentah bergantung ukuran perusahaan → rasio. Tunjukkan arah teori (hipotesis, bukan estimasi). M-Score dan Altman hanya **benchmark**. |
| 3:00 | **Prediksi Fraud** | Pilih Random Forest, **geser slider ambang**: jumlah perusahaan ditandai berubah seketika. Ambang default 9,1% berasal dari biaya FN:FP = 10:1. Klik satu baris → gauge 3D. |
| 3:50 | **Analisis Perusahaan** | *Analisis perusahaan* → faktor utama (Shapley eksak). Geser slider DSRI: probabilitas naik; tunjukkan peringatan ekstrapolasi. |
| 4:40 | **Penjelasan Model** | Forest plot odds ratio (mis. DSRI ×2,0 per 1 SD) dan kesesuaian dengan teori. SHAP: model nonlinear menangkap efek ambang pada `round_share_1000`. |
| 5:30 | **Evaluasi Model** | CI bootstrap **lebar** (hanya 43 fraud di data uji). Kalibrasi baik. DeLong: **tidak ada pemenang** antar model terawasi → alasan memilih regresi logistik yang dapat ditafsirkan. |
| 6:15 | **Laporan PDF** | **Buat laporan PDF** → unduh. Tutup dengan halaman **Tentang Model**: asumsi, keterbatasan, bias. |

## Pertanyaan yang mungkin diajukan dosen
- **"Mengapa tidak memakai accuracy?"** Fraud ±6%: model yang selalu menjawab "tidak fraud" sudah 94% akurat. Gunakan PR-AUC, recall, dan kalibrasi.
- **"Apakah ada data leakage?"** Pembagian per perusahaan (satu perusahaan tidak muncul di latih dan uji), resampling dan preprocessing di dalam pipeline, nested CV; uji permutasi label membuktikan AUC ≈ 0,5 pada data tanpa sinyal. Seleksi fitur dilakukan sebelum CV, sehingga skor CV sedikit optimistis; gunakan hold-out.
- **"Mengapa model terbaik tidak ditentukan?"** Selisih PR-AUC 0,02–0,03 dengan 43 fraud berada dalam derau (p Holm 0,74–0,84).
- **"Datanya asli?"** Tidak. Data sintetis hanya untuk mendemonstrasikan sistem; fitur pembulatan digit sengaja kuat. Hasil substantif skripsi harus berasal dari data asli (ganti file input; skema ada di `src/data/schema.py`).
- **"Bagaimana probabilitas bisa dipercaya?"** Platt scaling dilatih pada prediksi out-of-fold; slope kalibrasi 0,9–1,3, ECE < 0,02.
