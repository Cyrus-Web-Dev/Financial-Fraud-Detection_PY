# Sistem Deteksi Financial Fraud (Skripsi Statistika)

Aplikasi web end-to-end untuk mendeteksi **indikasi** fraud pada laporan keuangan, dengan fokus pada ketepatan metodologi statistik,
interpretabilitas, dan kejujuran ilmiah. Skor risiko adalah indikasi untuk diperiksa lebih lanjut, **bukan bukti fraud**.

**Stack:** Python (FastAPI, Scikit-Learn, SciPy, ReportLab, matplotlib) · React 19 + TypeScript + Vite + Zustand + Framer Motion + GSAP + Three.js.

> **Penting:** data pada `data/sample/` adalah **sintetis** (untuk pengembangan dan demo). Untuk skripsi, ganti dengan data perusahaan asli
> yang dipetakan ke skema di `src/data/schema.py`. Fitur pembulatan digit sengaja dibuat kuat pada generator; pada data nyata efeknya jauh lebih kecil.

## Mulai cepat
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m scripts.run_pipeline --quick                 # data -> EDA -> fitur -> model -> evaluasi -> penjelasan -> laporan (±6 menit)
python -m tests.run_all                                # 171 uji Python (atau: pytest tests -q)

uvicorn src.api.main:app --port 8000                   # terminal 1: API (dokumentasi: http://localhost:8000/docs)
cd frontend && npm install && npm run dev              # terminal 2: http://localhost:5173
cd frontend && npm test && npm run typecheck           # 41 uji logika + tsc
cd frontend && npm run build                           # lalu cukup `uvicorn ...`: API + frontend satu proses
```
Panduan demo 5–7 menit: [`DEMO.md`](DEMO.md). Laporan PDF contoh: `reports/contoh_laporan.pdf`.

## Tahapan dan modul
| Tahap | Isi | Lokasi |
|---|---|---|
| 1 | Konfigurasi, skema kolom, generator data panel sintetis (fraud sebagai manipulasi akrual) | `src/config.py`, `src/data/schema.py`, `src/data/synthetic.py` |
| 2 | Loader CSV/XLSX, validasi, outlier bebas-skala, EDA, uji Benford (digit pertama/terakhir) | `src/data/`, `src/evaluation/{eda,benford}.py` |
| 3 | 36 fitur (8 indeks Beneish), M-Score, Altman Z, VIF, Lasso + stability selection, split per perusahaan | `src/features/`, `src/data/split.py` |
| 4 | Regresi logistik (IRLS + Firth), Random Forest, Gradient Boosting, Isolation Forest; resampling anti-leakage; nested CV | `src/models/` |
| 5 | Kalibrasi, bootstrap per perusahaan, DeLong, McNemar, ambang berbasis biaya, decision curve, diagnostik | `src/evaluation/` |
| 6 | Odds ratio + interpretasi teori, Shapley eksak, permutation importance, penjelasan lokal | `src/explanation/` |
| 7 | Layanan Python murni (satu method per endpoint) + FastAPI tipis, riwayat SQLite | `src/services/`, `src/api/` |
| 8 | Frontend React (11 halaman, grafik SVG buatan sendiri, hero dan gauge 3D) | `frontend/` |
| 9 | Laporan PDF, pipeline satu perintah, uji menyeluruh, panduan demo | `src/reporting/`, `scripts/` |

Skrip: `run_pipeline` (semua tahap) · `run_eda` · `run_feature_engineering` · `run_modeling` · `run_evaluation` · `run_explanation` ·
`make_report` (PDF) · `export_frontend_fixtures`. Tiap skrip dapat dijalankan sendiri: `python -m scripts.<nama>`.

## API
| Metode | Rute | Fungsi |
|---|---|---|
| GET | `/api/health`, `/api/models`, `/api/models/{nama}`, `/api/features/catalog` | Informasi sistem |
| POST | `/api/datasets`, `/api/datasets/sample` | Unggah + validasi; data sampel |
| GET/DELETE | `/api/datasets/{id}/preview`, `/eda`, `/features`; `DELETE /api/datasets/{id}` | Pratinjau, EDA, rasio |
| POST | `/api/predict/dataset`, `/api/predict/rows`, `/api/what-if` | Prediksi batch/tunggal dan what-if |
| POST | `/api/report` | Laporan PDF |
| GET/DELETE | `/api/results/{jenis}`, `/api/history` | Hasil analisis; riwayat prediksi |

Format error seragam: `{"error": {"kode": "...", "pesan": "...", "rincian": ...}}`. Batas unggah 20 MB, prediksi 5.000 baris per permintaan,
penjelasan per baris untuk ≤ 50 baris; `dataset_id` divalidasi regex; query SQL berparameter; detail galat internal tidak dikirim ke klien.

## Hasil pada data sampel (hold-out: 720 baris, 43 fraud)
| Model | PR-AUC | ROC-AUC |
|---|---|---|
| Random Forest | 0,549 | 0,882 |
| Gradient Boosting | 0,543 | 0,871 |
| Regresi Logistik | 0,521 | 0,868 |
| Isolation Forest | 0,322 | 0,762 |
| Beneish M-Score (benchmark) | 0,220 | 0,799 |

CI 95% lebar, dan **tidak ada perbedaan signifikan antar model terawasi** (DeLong, Holm p ≥ 0,74). Jangan menyimpulkan "model terbaik";
regresi logistik setara dan jauh lebih mudah ditafsirkan.

## Keputusan metodologis yang perlu dijelaskan di skripsi
- **Anti-leakage:** split per perusahaan (data panel); resampling dan preprocessing di dalam `Pipeline`; uji permutasi label membuktikan AUC ≈ 0,5 tanpa sinyal.
- **Seleksi fitur** dilakukan sebelum CV pada data latih, sehingga skor CV sedikit optimistis; angka akhir adalah hold-out.
- **Kalibrasi:** Platt scaling pada prediksi out-of-fold; ambang default = ambang Bayes dari rasio biaya FN:FP = 10:1 (asumsi, dapat diubah).
- **Inferensi:** bootstrap per perusahaan (data panel); DeLong mengasumsikan independensi sehingga dilengkapi bootstrap berpasangan.
- **Shapley eksak** (enumerasi 2^d koalisi, d ≤ 14) dengan latar 40 baris latih: eksak terhadap latar tersebut; definisi sama dengan
  `shap.TreeExplainer(..., feature_perturbation="interventional")`.
- Arah teori pada katalog fitur adalah arah **hipotesis** (mis. Beneish 1999), bukan tanda koefisien estimasi pada sampel AS.

## Penyimpangan dari rencana awal
- Laporan PDF memakai **ReportLab** (bukan fpdf2) dan **matplotlib** untuk grafik; font DejaVu Sans disertakan (`src/reporting/fonts/`, lisensi bebas).
- Riwayat prediksi memakai `sqlite3` bawaan Python (bukan SQLAlchemy); validasi dengan pandas murni (bukan Pandera).
- Frontend memakai CSS biasa (bukan Tailwind) dan grafik SVG buatan sendiri (bukan Plotly).
- Regresi logistik inferensial dan Shapley ditulis sendiri (diverifikasi terhadap scikit-learn, Hessian numerik, brute force, dan rumus analitik);
  uji perbandingan dengan Statsmodels dan `shap` berjalan otomatis bila pustakanya terpasang.

## Pengujian
`python -m tests.run_all` (171 uji) mencakup generator data, validasi, Benford, fitur, seleksi, resampling, regresi logistik, evaluasi, diagnostik,
Shapley, layanan, API (statis dengan AST; end-to-end `TestClient` bila FastAPI terpasang), laporan PDF, dan orkestrasi pipeline.
Frontend: `npm test` (41 uji logika, termasuk kontrak terhadap JSON asli backend) dan `npm run typecheck`.

## Keterbatasan
Data sintetis; fraud pada data uji sedikit sehingga CI lebar; label fraud nyata rentan selection dan measurement bias; model tidak menjamin generalisasi
ke sektor/periode lain. Lihat halaman **Tentang Model** dan bagian 7 laporan PDF.

Font DejaVu Sans © Bitstream Vera / DejaVu (lisensi: `src/reporting/fonts/LICENSE-DejaVu.txt`).
