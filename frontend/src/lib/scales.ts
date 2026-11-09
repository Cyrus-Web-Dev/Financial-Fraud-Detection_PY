// Skala dan utilitas grafik SVG (fungsi murni, mudah diuji).

export type Scale = (v: number) => number;

export function linearScale(d0: number, d1: number, r0: number, r1: number): Scale {
  if (d0 === d1) return () => (r0 + r1) / 2;
  const k = (r1 - r0) / (d1 - d0);
  return (v) => r0 + (v - d0) * k;
}

export function extent(values: Iterable<number | null | undefined>): [number, number] | null {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of values) {
    if (typeof v === "number" && Number.isFinite(v)) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  return lo <= hi ? [lo, hi] : null;
}

/** Tick "cantik" (kelipatan 1, 2, 5 x 10^k) yang mencakup [min, max]. */
export function niceTicks(min: number, max: number, count = 5): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [];
  if (min > max) [min, max] = [max, min];
  if (min === max) return [min];
  const rawStep = (max - min) / Math.max(1, count - 1);
  const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const norm = rawStep / mag;
  const step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10) * mag;
  const start = Math.floor(min / step + 1e-9) * step;
  const end = Math.ceil(max / step - 1e-9) * step;
  const decimals = Math.max(0, -Math.floor(Math.log10(step)) + 1);
  const ticks: number[] = [];
  for (let v = start; v <= end + step * 1e-9; v += step) ticks.push(Number(v.toFixed(decimals)));
  return ticks;
}

/** Tick "cantik" yang DIBATASI di dalam [d0, d1] (untuk domain sumbu yang ditentukan eksplisit). */
export function ticksIn(d0: number, d1: number, count = 5): number[] {
  const lo = Math.min(d0, d1);
  const hi = Math.max(d0, d1);
  const eps = (hi - lo) * 1e-9;
  return niceTicks(lo, hi, count).filter((t) => t >= lo - eps && t <= hi + eps);
}

export function niceDomain(min: number, max: number, count = 5): [number, number] {
  const t = niceTicks(min, max, count);
  if (t.length === 0) return [0, 1];
  const lo = t[0] as number;
  const hi = t[t.length - 1] as number;
  return lo === hi ? [lo - 1, hi + 1] : [lo, hi];
}

/** Path SVG garis; nilai null/NaN memutus garis (segmen baru). */
export function linePath(points: ReadonlyArray<readonly [number | null, number | null]>): string {
  let d = "";
  let pen = false;
  for (const [x, y] of points) {
    if (typeof x !== "number" || typeof y !== "number" || !Number.isFinite(x) || !Number.isFinite(y)) {
      pen = false;
      continue;
    }
    d += `${pen ? "L" : "M"}${round(x)} ${round(y)}`;
    pen = true;
  }
  return d;
}

export function round(v: number, digits = 2): number {
  const k = Math.pow(10, digits);
  return Math.round(v * k) / k;
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function mix(c0: readonly [number, number, number], c1: readonly [number, number, number], t: number): string {
  return `rgb(${Math.round(lerp(c0[0], c1[0], t))}, ${Math.round(lerp(c0[1], c1[1], t))}, ${Math.round(lerp(c0[2], c1[2], t))})`;
}

const BLUE: readonly [number, number, number] = [47, 111, 222];
const WHITE: readonly [number, number, number] = [255, 255, 255];
const RED: readonly [number, number, number] = [215, 38, 61];

/** Skala divergen biru - putih - merah untuk nilai di [vmin, vmax] (mis. korelasi -1..1). */
export function divergingColor(v: number | null | undefined, vmin = -1, vmax = 1): string {
  if (typeof v !== "number" || !Number.isFinite(v)) return "rgb(240, 242, 247)";
  const mid = (vmin + vmax) / 2;
  const t = Math.min(1, Math.abs(v - mid) / ((vmax - vmin) / 2));
  return v < mid ? mix(WHITE, BLUE, t) : mix(WHITE, RED, t);
}

/** Skala berurutan putih -> biru tua untuk t di [0, 1]. */
export function sequentialColor(t: number): string {
  const c = Math.min(1, Math.max(0, Number.isFinite(t) ? t : 0));
  return mix([232, 240, 252], [11, 31, 58], c);
}

/** Warna titik beeswarm: persentil nilai fitur rendah (biru) -> tinggi (merah). */
export function percentileColor(t: number): string {
  const c = Math.min(1, Math.max(0, Number.isFinite(t) ? t : 0.5));
  return c < 0.5 ? mix(BLUE, [150, 120, 200], c * 2) : mix([150, 120, 200], RED, (c - 0.5) * 2);
}

/** Kuantil sederhana (interpolasi linear) pada data tak terurut. */
export function quantile(values: number[], q: number): number {
  const v = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (v.length === 0) return NaN;
  const pos = (v.length - 1) * Math.min(1, Math.max(0, q));
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return (v[lo] as number) + ((v[hi] as number) - (v[lo] as number)) * (pos - lo);
}

/** Potong label agar muat dalam ``px`` piksel ruang SVG (perkiraan ~6,7 px per karakter; diukur dari render nyata). */
export function fitLabel(label: string, px: number): string {
  const max = Math.max(6, Math.floor(px / 6.7));
  return label.length > max ? `${label.slice(0, max - 1)}…` : label;
}
