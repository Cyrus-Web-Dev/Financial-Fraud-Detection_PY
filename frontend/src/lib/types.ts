// Tipe respons API. Bentuknya mengikuti keluaran layanan Python (lihat tests/fixtures/*.json).
// Hanya tipe (tanpa runtime) sehingga aman dibaca langsung oleh Node.

export type Severity = "error" | "warning" | "info";
export type RiskLevel = "Tinggi" | "Sedang" | "Rendah";
export type ModelName = "logit_mle" | "logreg" | "random_forest" | "hist_gb";
export type Row = Record<string, string | number | boolean | null>;

export type ApiErrorBody = { error: { kode: string; pesan: string; rincian?: unknown } };

export type Health = { status: string; siap_prediksi: boolean; model_tersedia: string[]; jumlah_riwayat: number };
export type ModelEntry = { nama: ModelName; label: string; jenis: string; satuan_penjelasan: string; tersedia: boolean };
export type ModelList = { model: ModelEntry[]; ambang_default: number | null };

export type FeatureRange = { fitur: string; label: string; rumus: string; kelompok: string; min: number; p1: number; median: number; p99: number; max: number };
export type CatalogItem = { name: string; label: string; group: string; formula: string; rationale: string; expected_sign: number; dipakai_model: boolean };
export type Catalog = { fitur: CatalogItem[]; fitur_terpilih: string[] };
export type ModelDetail = {
  nama: string; label: string; jenis: string; satuan_penjelasan: string; metadata: Record<string, unknown>;
  metrik_uji?: Record<string, number | string | null>; rentang_fitur: FeatureRange[]; interpretasi_odds_ratio?: OddsInterp[] | null;
};

export type Issue = { severity: Severity; code: string; message: string; column: string | null; count: number | null; share: number | null };
export type ValidationReport = {
  is_valid: boolean; n_errors: number; n_warnings: number; issues: Issue[];
  summary: { n_baris: number; n_kolom: number; n_perusahaan?: number; rentang_tahun?: [number, number]; [k: string]: unknown };
};
export type UploadResult = { dataset_id: string; nama: string; n_baris: number; punya_label: boolean; validasi: ValidationReport; pratinjau: Row[] };
export type Preview = { meta: { id: string; nama: string; n_baris: number; n_kolom: number; punya_label?: boolean }; offset: number; total: number; baris: Row[] };
export type FeatureTable = { fitur: string[]; baris: Row[] };

export type Factor = { fitur: string; label: string; nilai: number; kontribusi: number; arah: string; satuan: string };
export type PredictionRow = {
  company_id: string; tahun: number; sektor: string; skor_model: number; probabilitas: number; ditandai: boolean;
  tingkat_risiko: RiskLevel; m_score: number | null; m_flag: boolean | null; altman_z2_zona: string | null; faktor_utama?: Factor[];
};
export type PredictionResult = {
  model: ModelName; label_model: string; ambang: number; n_baris: number; n_ditandai: number; persen_ditandai: number;
  nilai_dasar_penjelasan: number | null; satuan_penjelasan: string; catatan: string[]; hasil: PredictionRow[];
  dataset_id?: string; riwayat_tersimpan: number;
};
export type WhatIfState = { probabilitas: number; ditandai: boolean; tingkat_risiko: RiskLevel };
export type WhatIfResult = {
  model: ModelName; ambang: number; sebelum: WhatIfState; sesudah: WhatIfState; selisih_probabilitas: number;
  perubahan: Record<string, { dari: number; ke: number }>; faktor_utama_sesudah: Factor[]; peringatan: string[];
};
export type HistoryRecord = {
  id: number; dibuat: string; model: string; sumber: string; company_id: string | null; tahun: number | null;
  probabilitas: number; ambang: number; ditandai: boolean; tingkat_risiko: RiskLevel; rincian: unknown;
};
export type HistoryList = { total: number; data: HistoryRecord[] };

export type PredictDatasetRequest = { dataset_id: string; model: ModelName; threshold?: number | null; explain?: boolean; tahun?: number | null };
export type PredictRowsRequest = { rows: Row[]; model: ModelName; threshold?: number | null; explain?: boolean };
export type ReportRequest = { dataset_id: string; model: ModelName; threshold?: number | null; tahun?: number | null; top_n?: number };
export type WhatIfRequest = { model: ModelName; features: Record<string, number>; overrides: Record<string, number>; threshold?: number | null };

// ---- EDA
export type DescRow = { kolom: string; n: number; missing: number; persen_missing: number; mean: number | null; sd: number | null; min: number | null; q1: number | null; median: number | null; q3: number | null; max: number | null; skewness: number | null; kurtosis: number | null };
export type NormalityRow = { kolom: string; n: number; shapiro_W: number | null; shapiro_p: number | null; dagostino_p: number | null; normal: string };
export type DigitTest = {
  kind: string; digits: number[]; n: number; observed_count: number[]; observed_prop: (number | null)[]; expected_prop: number[];
  z_scores: (number | null)[]; chi2: number | null; dof: number; p_value: number | null; mad: number | null; conformity: string | null; small_sample: boolean;
};
export type BenfordGroup = { n_baris: number; first_digit: DigitTest; last_digit: DigitTest };
export type ClassDist = { kelompok: string; n: number; fraud: number; non_fraud: number; proporsi_fraud: number | null; ci95_bawah: number | null; ci95_atas: number | null };
export type GroupCompare = { variabel: string; n_fraud: number; n_non_fraud: number; median_fraud: number | null; median_non_fraud: number | null; U: number | null; p_value: number | null; efek_rank_biserial: number | null; q_value: number | null };
export type EdaSummary = {
  deskriptif: DescRow[]; normalitas: NormalityRow[]; korelasi: { kolom: string[]; matriks: (number | null)[][] };
  korelasi_tinggi: { variabel_1: string; variabel_2: string; korelasi: number }[];
  outlier: { kolom: string; jumlah_outlier: number; persen: number }[];
  benford_per_kolom: { kolom: string; n: number; mad: number | null; chi2: number | null; p_value: number | null; kesesuaian: string | null }[];
  benford_per_kelompok: Record<string, BenfordGroup>;
  distribusi_kelas?: ClassDist[]; distribusi_per_tahun?: ClassDist[]; distribusi_per_sektor?: ClassDist[]; perbandingan_kelompok?: GroupCompare[];
};

// ---- Evaluasi
export type BootRow = { model: string; metrik: string; estimasi: number; ci_bawah: number; ci_atas: number; se_bootstrap: number; n_replikasi_valid: number };
export type CalibRow = { model: string; versi: string; brier: number; brier_skill_score: number | null; ece: number; slope: number; slope_ci_bawah: number; slope_ci_atas: number; intersep: number; citl: number; rata_prediksi: number; frekuensi_teramati: number };
export type CalibBin = { bin: number; n: number; rata_prediksi: number; frekuensi_teramati: number; ci95_bawah: number; ci95_atas: number };
export type ThresholdRow = {
  model: string; aturan: string; threshold: number; tp: number; fp: number; tn: number; fn: number; precision: number | null; recall: number | null;
  specificity: number | null; f1: number | null; accuracy: number | null; balanced_accuracy: number | null; biaya_total: number; biaya_per_observasi: number;
  biaya_tanpa_model: number; biaya_tanpa_model_per_observasi: number; penghematan_biaya_persen: number | null; persen_ditandai: number;
};
export type PairRow = {
  model_a: string; model_b: string; auc_a: number; auc_b: number; selisih_auc: number; p_delong: number; selisih_pr_auc: number;
  pr_ci_bawah: number; pr_ci_atas: number; p_bootstrap_pr: number; p_delong_holm: number; p_bootstrap_pr_holm: number;
};
export type McNemarRow = { model_a: string; model_b: string; n_fraud: number; b: number; c: number; statistik: number; p_value: number; metode: string };
export type DcaRow = { ambang: number; net_benefit_model: number; net_benefit_tandai_semua: number; net_benefit_tidak_ada: number };
export type CurvePack = { roc: { fpr: number[]; tpr: number[] }; pr: { recall: number[]; precision: number[] } };
export type LogitDiagnostics = {
  hosmer_lemeshow_latih: { statistik: number; dof: number; p_value: number }; hosmer_lemeshow_uji: { statistik: number; dof: number; p_value: number };
  tabel_hl_latih: { kelompok: number; n: number; fraud_teramati: number; fraud_diharapkan: number; rata_prob: number }[];
  box_tidwell: { fitur: string; koef_xlnx: number; z: number; p_value: number; ambang_bonferroni: number; linear: boolean; digeser: boolean }[];
  pengaruh_ringkasan: Record<string, number>; pengaruh_teratas: Record<string, number | string>[]; vif: Record<string, number>[] | Record<string, number>;
  catatan_residual: string; kecocokan_model: Record<string, number | boolean>;
};
export type EvaluationResults = {
  biaya: { c_fn: number; c_fp: number; ambang_bayes: number }; n_uji: number; fraud_uji: number; n_bootstrap: number;
  metrik_uji_bootstrap: BootRow[]; auc_delong_ci: Record<string, { auc: number; se: number; ci_bawah: number; ci_atas: number }>;
  kalibrasi: CalibRow[]; kurva_kalibrasi: Record<string, CalibBin[]>; ambang_keputusan: ThresholdRow[]; perbandingan_model: PairRow[];
  mcnemar_sensitivitas_top10: McNemarRow[]; decision_curve: Record<string, DcaRow[]>; dca_proporsi_ambang_berguna: Record<string, number>;
  kurva: Record<string, CurvePack>; diagnostik_logit: LogitDiagnostics;
};

// ---- Penjelasan
export type OddsInterp = {
  fitur: string; label: string; odds_ratio: number; ci_bawah: number; ci_atas: number; p_value: number; signifikan: boolean;
  arah_teramati: string; arah_teori: string; kesesuaian: string; teori: string; interpretasi: string;
};
export type ShapGlobalRow = { fitur: string; mean_abs_shap: number; rata_bertanda: number; porsi: number; peringkat: number };
export type BeeswarmPoint = { fitur: string; baris: number; nilai_fitur: number; persentil_nilai: number; shap: number };
export type DependenceData = { fitur: string; warna_fitur: string | null; x: number[]; shap: number[]; nilai_warna?: number[] };
export type PermRow = { fitur: string; penurunan_mean: number; penurunan_sd: number | null; metrik_dasar: number };
export type LocalExplanation = {
  nilai_dasar: number; prediksi: number; satuan: string; kalimat: string[];
  kontribusi: { fitur: string; label: string; nilai: number; persentil_latih: number | null; kontribusi: number }[];
};
export type LocalCase = { company_id: string; tahun: number; label_sebenarnya: number; random_forest: LocalExplanation; regresi_logistik: LocalExplanation };
export type ExplanationResults = {
  fitur: string[]; n_dijelaskan: number; n_latar: number; catatan_sampel: string;
  interpretasi_odds_ratio: OddsInterp[]; mle_vs_firth: { fitur: string; mle: number; firth: number; selisih: number }[];
  shap_global: Record<string, ShapGlobalRow[]>; shap_nilai_dasar: Record<string, number>; shap_satuan: Record<string, string>;
  beeswarm: Record<string, BeeswarmPoint[]>; dependence: Record<string, Record<string, DependenceData>>;
  permutation_importance: Record<string, PermRow[]>;
  kesepakatan_peringkat: { spearman: Record<string, number>[]; irisan_top3: Record<string, number>[]; top_fitur: Record<string, string[]> };
  penjelasan_lokal: LocalCase[];
  perbandingan_interpretabilitas: { model: string; tingkat: string; bentuk_penjelasan: string; catatan: string }[];
};

// ---- Modeling
export type HoldoutRow = { model: string; label: string; strategi: string; roc_auc: number; pr_auc: number; recall_at_10: number; precision_at_10: number };
export type ModelingResults = {
  fitur_terpilih: string[]; split: Record<string, number | number[]>; versi_library: Record<string, string>;
  distribusi_kelas_resampling: { strategi: string; non_fraud_sebelum: number; fraud_sebelum: number; non_fraud_sesudah: number; fraud_sesudah: number; proporsi_fraud_sesudah: number; catatan: string }[];
  perbandingan_imbalance: { model: string; strategi: string; roc_auc_mean: number; roc_auc_sd: number; pr_auc_mean: number; pr_auc_sd: number; recall_at_10_mean: number }[];
  nested_cv: Record<string, { strategi: string; parameter_final: Record<string, unknown>; pr_auc_mean: number; pr_auc_sd: number; roc_auc_mean: number; recall_at_10_mean: number }>;
  uji_holdout: HoldoutRow[]; uji_temporal: { model: string; roc_auc: number; pr_auc: number; recall_at_10: number; precision_at_10: number }[];
  ablasi_fitur?: { model: string; himpunan_fitur: string; n_fitur: number; pr_auc_mean: number; pr_auc_sd: number; roc_auc_mean: number }[];
};
export type ResultKind = "modeling" | "evaluation" | "explanation" | "feature_selection" | "eda_sample";
