// Router berbasis hash (#/halaman) tanpa dependensi tambahan.
export type PageId = "beranda" | "unggah" | "eda" | "fitur" | "prediksi" | "perusahaan" | "penjelasan" | "evaluasi" | "riwayat" | "laporan" | "tentang";

export type PageDef = { id: PageId; label: string; group: string; hint: string; needsDataset: boolean };

export const PAGES: readonly PageDef[] = [
  { id: "beranda", label: "Beranda", group: "Mulai", hint: "Ringkasan sistem", needsDataset: false },
  { id: "unggah", label: "Unggah & Validasi", group: "Data", hint: "Unggah data dan periksa kualitasnya", needsDataset: false },
  { id: "eda", label: "Eksplorasi Data", group: "Data", hint: "Statistik deskriptif, korelasi, Benford", needsDataset: true },
  { id: "fitur", label: "Feature Engineering", group: "Data", hint: "Rasio keuangan, M-Score, Z-Score", needsDataset: true },
  { id: "prediksi", label: "Prediksi Fraud", group: "Model", hint: "Prediksi batch dengan ambang", needsDataset: true },
  { id: "perusahaan", label: "Analisis Perusahaan", group: "Model", hint: "Penjelasan dan what-if per perusahaan", needsDataset: true },
  { id: "penjelasan", label: "Penjelasan Model", group: "Model", hint: "Odds ratio dan SHAP", needsDataset: false },
  { id: "evaluasi", label: "Evaluasi Model", group: "Model", hint: "Metrik, kalibrasi, perbandingan", needsDataset: false },
  { id: "riwayat", label: "Riwayat", group: "Sistem", hint: "Riwayat prediksi", needsDataset: false },
  { id: "laporan", label: "Laporan PDF", group: "Sistem", hint: "Ekspor laporan analisis", needsDataset: true },
  { id: "tentang", label: "Tentang Model", group: "Sistem", hint: "Asumsi dan keterbatasan", needsDataset: false },
];

export const DEFAULT_PAGE: PageId = "beranda";

export function isPageId(x: string): x is PageId {
  return PAGES.some((p) => p.id === x);
}

/** "#/prediksi" -> "prediksi"; hash tak dikenal -> halaman default. */
export function pageFromHash(hash: string): PageId {
  const id = hash.replace(/^#\/?/, "").split(/[/?]/)[0] ?? "";
  return isPageId(id) ? id : DEFAULT_PAGE;
}

export function hashFor(id: PageId): string {
  return `#/${id}`;
}

export function pageDef(id: PageId): PageDef {
  return PAGES.find((p) => p.id === id) ?? (PAGES[0] as PageDef);
}

export function groupedPages(): { group: string; pages: PageDef[] }[] {
  const out: { group: string; pages: PageDef[] }[] = [];
  for (const p of PAGES) {
    let g = out.find((x) => x.group === p.group);
    if (!g) out.push((g = { group: p.group, pages: [] }));
    g.pages.push(p);
  }
  return out;
}
