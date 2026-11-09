// Logika state murni (tanpa React/zustand) sehingga dapat diuji langsung.
import type { Issue, PredictionRow, Row, Severity, FeatureRange, RiskLevel } from "../lib/types.ts";
import { isNum } from "../lib/format.ts";

export type PredictionSummary = { total: number; tinggi: number; sedang: number; rendah: number; flagged: number; meanProb: number; maxProb: number };

export function summarizePredictions(rows: readonly PredictionRow[]): PredictionSummary {
  const s: PredictionSummary = { total: rows.length, tinggi: 0, sedang: 0, rendah: 0, flagged: 0, meanProb: 0, maxProb: 0 };
  let sum = 0;
  for (const r of rows) {
    if (r.tingkat_risiko === "Tinggi") s.tinggi++;
    else if (r.tingkat_risiko === "Sedang") s.sedang++;
    else s.rendah++;
    if (r.ditandai) s.flagged++;
    sum += r.probabilitas;
    if (r.probabilitas > s.maxProb) s.maxProb = r.probabilitas;
  }
  s.meanProb = rows.length ? sum / rows.length : 0;
  return s;
}

export type PredictionFilter = { query: string; onlyFlagged: boolean; risk: RiskLevel | "semua" };

export function filterPredictions(rows: readonly PredictionRow[], f: PredictionFilter): PredictionRow[] {
  const q = f.query.trim().toLowerCase();
  return rows.filter((r) =>
    (!f.onlyFlagged || r.ditandai) &&
    (f.risk === "semua" || r.tingkat_risiko === f.risk) &&
    (q === "" || r.company_id.toLowerCase().includes(q) || r.sektor.toLowerCase().includes(q) || String(r.tahun).includes(q)));
}

export type SortDir = "asc" | "desc";

/** Urutkan salinan; nilai null/NaN selalu di akhir; pengurutan stabil. */
export function sortRows<T>(rows: readonly T[], get: (r: T) => string | number | null | undefined, dir: SortDir = "asc"): T[] {
  const idx = rows.map((r, i) => ({ r, i, v: get(r) }));
  idx.sort((a, b) => {
    const an = a.v === null || a.v === undefined || (typeof a.v === "number" && !Number.isFinite(a.v));
    const bn = b.v === null || b.v === undefined || (typeof b.v === "number" && !Number.isFinite(b.v));
    if (an || bn) return an === bn ? a.i - b.i : an ? 1 : -1;
    const c = typeof a.v === "number" && typeof b.v === "number" ? a.v - b.v : String(a.v).localeCompare(String(b.v), "id");
    return c !== 0 ? (dir === "asc" ? c : -c) : a.i - b.i;
  });
  return idx.map((x) => x.r);
}

export function paginate<T>(rows: readonly T[], page: number, pageSize: number): { items: T[]; page: number; pages: number; total: number } {
  const size = Math.max(1, Math.floor(pageSize));
  const pages = Math.max(1, Math.ceil(rows.length / size));
  const p = Math.min(pages, Math.max(1, Math.floor(page)));
  return { items: rows.slice((p - 1) * size, p * size), page: p, pages, total: rows.length };
}

export function groupIssues(issues: readonly Issue[]): Record<Severity, Issue[]> {
  const out: Record<Severity, Issue[]> = { error: [], warning: [], info: [] };
  for (const i of issues) out[i.severity].push(i);
  return out;
}

export function uniqueSorted<T extends string | number>(values: Iterable<T>): T[] {
  return [...new Set(values)].sort((a, b) => (typeof a === "number" && typeof b === "number" ? a - b : String(a).localeCompare(String(b))));
}

export function companyLabel(r: Row): string {
  return `${String(r["company_id"] ?? "?")} · ${String(r["tahun"] ?? "?")} · ${String(r["sektor"] ?? "-")}`;
}

/** Ambil nilai fitur model dari satu baris tabel fitur; null bila ada yang kosong/tak terhingga. */
export function featureBase(row: Row, selected: readonly string[]): Record<string, number> | null {
  const out: Record<string, number> = {};
  for (const f of selected) {
    const v = row[f];
    if (!isNum(v)) return null;
    out[f] = v;
  }
  return out;
}

/** Langkah slider ~1/200 rentang p1-p99, dibulatkan ke 1 angka penting. */
export function sliderStep(r: Pick<FeatureRange, "p1" | "p99">): number {
  const span = r.p99 - r.p1;
  if (!(span > 0)) return 0.01;
  const raw = span / 200;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  return Math.max(mag, Math.round(raw / mag) * mag);
}

/** Batas slider: sedikit melebihi p1-p99 agar nilai ekstrem tetap dapat dicoba. */
export function sliderBounds(r: Pick<FeatureRange, "p1" | "p99" | "min" | "max">): [number, number] {
  const span = r.p99 - r.p1;
  const pad = span > 0 ? span * 0.25 : 1;
  return [Math.max(r.min - pad * 0.1, r.p1 - pad), Math.min(r.max + pad * 0.1, r.p99 + pad)];
}

export function isOutsideTrainingRange(v: number, r: Pick<FeatureRange, "min" | "max">): boolean {
  return v < r.min || v > r.max;
}

export function thresholdMarks(defaultThreshold: number | null): { value: number; label: string }[] {
  const marks = [{ value: 0.05, label: "5%" }, { value: 0.2, label: "20%" }, { value: 0.5, label: "50%" }];
  if (defaultThreshold !== null && Number.isFinite(defaultThreshold)) marks.push({ value: defaultThreshold, label: "default" });
  return marks.sort((a, b) => a.value - b.value);
}

/** Aturan tingkat risiko yang SAMA dengan backend: Tinggi p >= ambang; Sedang p >= ambang/2; selainnya Rendah. */
export function riskLevel(p: number, threshold: number): RiskLevel {
  return p >= threshold ? "Tinggi" : p >= threshold / 2 ? "Sedang" : "Rendah";
}

/** Klasifikasi ulang hasil prediksi pada ambang baru tanpa memanggil API (probabilitas tidak berubah). */
export function reclassify(rows: readonly PredictionRow[], threshold: number): PredictionRow[] {
  return rows.map((r) => ({ ...r, ditandai: r.probabilitas >= threshold, tingkat_risiko: riskLevel(r.probabilitas, threshold) }));
}

const HISTOGRAM_EDGES = [0, 0.02, 0.05, 0.1, 0.2, 0.4, 1.0000001] as const;

/** Histogram probabilitas dengan batas tetap (skala tidak linear karena fraud jarang). */
export function probabilityHistogram(rows: readonly PredictionRow[]): { label: string; value: number }[] {
  const labels = ["< 2%", "2–5%", "5–10%", "10–20%", "20–40%", "≥ 40%"];
  const counts = labels.map(() => 0);
  for (const r of rows) {
    for (let i = 0; i < counts.length; i++) {
      if (r.probabilitas >= (HISTOGRAM_EDGES[i] as number) && r.probabilitas < (HISTOGRAM_EDGES[i + 1] as number)) {
        counts[i] = (counts[i] as number) + 1;
        break;
      }
    }
  }
  return labels.map((label, i) => ({ label, value: counts[i] as number }));
}
