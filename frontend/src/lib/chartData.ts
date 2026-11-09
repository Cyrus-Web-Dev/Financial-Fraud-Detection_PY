// Transformasi hasil API menjadi struktur siap-gambar untuk komponen grafik.
import type {
  BeeswarmPoint, BootRow, CalibBin, ClassDist, CurvePack, DcaRow, DigitTest, EdaSummary, OddsInterp, ShapGlobalRow,
} from "./types.ts";
import { isNum } from "./format.ts";

export type Bar = { label: string; value: number; lo?: number; hi?: number; key?: string };
export type Series = { name: string; points: [number, number][] };

export function benfordBars(t: DigitTest): { digit: number; observed: number; expected: number; z: number | null }[] {
  return t.digits.map((d, i) => ({
    digit: d,
    observed: t.observed_prop[i] ?? 0,
    expected: t.expected_prop[i] ?? 0,
    z: t.z_scores[i] ?? null,
  }));
}

export function heatmapCells(k: EdaSummary["korelasi"]): { labels: string[]; values: (number | null)[][] } {
  return { labels: k.kolom, values: k.matriks };
}

/** Proporsi fraud per kelompok + CI Wilson (urut sesuai data). */
export function classBars(rows: ClassDist[]): Bar[] {
  return rows.map((r) => {
    const b: Bar = { label: r.kelompok, value: r.proporsi_fraud ?? 0 };
    if (isNum(r.ci95_bawah)) b.lo = r.ci95_bawah;
    if (isNum(r.ci95_atas)) b.hi = r.ci95_atas;
    return b;
  });
}

export function rocSeries(kurva: Record<string, CurvePack>, models: string[]): Series[] {
  return models.flatMap((m) => {
    const c = kurva[m];
    return c ? [{ name: m, points: c.roc.fpr.map((x, i): [number, number] => [x, c.roc.tpr[i] ?? 0]) }] : [];
  });
}

export function prSeries(kurva: Record<string, CurvePack>, models: string[]): Series[] {
  return models.flatMap((m) => {
    const c = kurva[m];
    return c ? [{ name: m, points: c.pr.recall.map((x, i): [number, number] => [x, c.pr.precision[i] ?? 0]) }] : [];
  });
}

export function calibrationPoints(bins: CalibBin[] | undefined): { x: number; y: number; lo: number; hi: number; n: number }[] {
  return (bins ?? []).map((b) => ({ x: b.rata_prediksi, y: b.frekuensi_teramati, lo: b.ci95_bawah, hi: b.ci95_atas, n: b.n }));
}

export function dcaSeries(rows: DcaRow[] | undefined): Series[] {
  const r = rows ?? [];
  return [
    { name: "Model", points: r.map((x): [number, number] => [x.ambang, x.net_benefit_model]) },
    { name: "Tandai semua", points: r.map((x): [number, number] => [x.ambang, x.net_benefit_tandai_semua]) },
    { name: "Tidak tandai siapa pun", points: r.map((x): [number, number] => [x.ambang, x.net_benefit_tidak_ada]) },
  ];
}

/** Kepentingan global SHAP sebagai persen (urut menurun), dengan label fitur. */
export function shapBars(rows: ShapGlobalRow[], labels: Record<string, string> = {}): Bar[] {
  return [...rows].sort((a, b) => b.mean_abs_shap - a.mean_abs_shap).map((r) => ({
    key: r.fitur, label: labels[r.fitur] ?? r.fitur, value: r.porsi * 100,
  }));
}

export function beeswarmByFeature(points: BeeswarmPoint[], order: string[]): { fitur: string; points: BeeswarmPoint[] }[] {
  return order.map((f) => ({ fitur: f, points: points.filter((p) => p.fitur === f) }));
}

export function oddsForest(rows: OddsInterp[]): { label: string; key: string; or: number; lo: number; hi: number; sig: boolean }[] {
  return rows.map((r) => ({ label: r.label, key: r.fitur, or: r.odds_ratio, lo: r.ci_bawah, hi: r.ci_atas, sig: r.signifikan }));
}

/** Satu metrik untuk semua model, lengkap dengan CI bootstrap. */
export function metricBars(boot: BootRow[], metric: string, order?: string[]): Bar[] {
  const rows = boot.filter((b) => b.metrik === metric);
  const sorted = order ? order.flatMap((m) => rows.filter((r) => r.model === m)) : rows;
  return sorted.map((r) => ({ key: r.model, label: r.model, value: r.estimasi, lo: r.ci_bawah, hi: r.ci_atas }));
}

/** Matriks kesepakatan (list of record) -> label + nilai. Urutan baris = urutan kolom. */
export function agreementMatrix(records: Record<string, number>[]): { labels: string[]; values: number[][] } {
  const first = records[0];
  const labels = first ? Object.keys(first) : [];
  return { labels, values: records.map((r) => labels.map((l) => r[l] ?? NaN)) };
}

export function symmetricLimit(values: number[]): number {
  const m = Math.max(0, ...values.filter(Number.isFinite).map(Math.abs));
  return m === 0 ? 1 : m;
}

/** Peringkat persentil (0-1) untuk pewarnaan; seri mendapat peringkat rata-rata. */
export function percentileRanks(values: readonly number[]): number[] {
  const idx = values.map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v);
  const out = new Array<number>(values.length).fill(0.5);
  if (values.length <= 1) return out;                      // satu nilai: tidak ada urutan -> warna netral
  let s = 0;
  while (s < idx.length) {
    let e = s;
    while (e + 1 < idx.length && (idx[e + 1] as { v: number }).v === (idx[s] as { v: number }).v) e++;
    const rank = (s + e) / 2 / Math.max(1, idx.length - 1);
    for (let k = s; k <= e; k++) out[(idx[k] as { i: number }).i] = rank;
    s = e + 1;
  }
  return out;
}
