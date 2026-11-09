// Pemformat angka bergaya Indonesia: pemisah ribuan titik, desimal koma.
import type { RiskLevel } from "./types.ts";

const DASH = "–";
const cache = new Map<string, Intl.NumberFormat>();

function nf(minDigits: number, maxDigits: number): Intl.NumberFormat {
  const key = `${minDigits}:${maxDigits}`;
  let f = cache.get(key);
  if (!f) {
    f = new Intl.NumberFormat("id-ID", { minimumFractionDigits: minDigits, maximumFractionDigits: maxDigits });
    cache.set(key, f);
  }
  return f;
}

export function isNum(x: unknown): x is number {
  return typeof x === "number" && Number.isFinite(x);
}

/** 1234.5 -> "1.234,50" (digits = jumlah desimal tetap). Nilai tak terdefinisi -> "–". */
export function formatNumber(x: number | null | undefined, digits = 2): string {
  return isNum(x) ? nf(digits, digits).format(x) : DASH;
}

export function formatInt(x: number | null | undefined): string {
  return isNum(x) ? nf(0, 0).format(Math.round(x)) : DASH;
}

/** Proporsi 0-1 -> persen: 0.0612 -> "6,1%". */
export function formatPercent(x: number | null | undefined, digits = 1): string {
  return isNum(x) ? `${nf(digits, digits).format(x * 100)}%` : DASH;
}

/** Probabilitas dengan presisi adaptif agar nilai kecil tidak tampak nol. */
export function formatProb(p: number | null | undefined): string {
  if (!isNum(p)) return DASH;
  if (p > 0 && p < 0.001) return "< 0,1%";
  if (p < 0.01) return formatPercent(p, 2);
  return formatPercent(p, 1);
}

export function formatPValue(p: number | null | undefined): string {
  if (!isNum(p)) return DASH;
  if (p < 0.001) return "< 0,001";
  return nf(3, 3).format(p);
}

/** Nilai uang dalam Rupiah JUTA -> ringkas: 1.500.000 -> "Rp 1,5 T". */
export function formatRupiahJuta(x: number | null | undefined): string {
  if (!isNum(x)) return DASH;
  const a = Math.abs(x);
  const sign = x < 0 ? "-" : "";
  if (a >= 1e6) return `${sign}Rp ${nf(1, 1).format(a / 1e6)} T`;
  if (a >= 1e3) return `${sign}Rp ${nf(1, 1).format(a / 1e3)} M`;
  return `${sign}Rp ${nf(0, 0).format(a)} jt`;
}

/** Format fleksibel: angka besar tanpa desimal, kecil dengan desimal bermakna. */
export function formatSmart(x: number | null | undefined): string {
  if (!isNum(x)) return DASH;
  const a = Math.abs(x);
  if (a === 0) return "0";
  if (a >= 1000) return formatInt(x);
  if (a >= 100) return formatNumber(x, 1);
  if (a >= 1) return formatNumber(x, 2);
  if (a >= 0.01) return formatNumber(x, 3);
  return x.toExponential(1).replace(".", ",");
}

export function clamp(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
}

export type Tone = "high" | "mid" | "low";
export function riskTone(level: RiskLevel | string): Tone {
  return level === "Tinggi" ? "high" : level === "Sedang" ? "mid" : "low";
}

/** Selisih probabilitas bertanda dalam poin persen: 0.0123 -> "+1,2 poin". */
export function formatDeltaPoints(d: number | null | undefined, digits = 1): string {
  if (!isNum(d)) return DASH;
  const s = d > 0 ? "+" : d < 0 ? "−" : "";
  return `${s}${nf(digits, digits).format(Math.abs(d) * 100)} poin`;
}

export const MODEL_LABELS: Record<string, string> = {
  logit_mle: "Regresi Logistik (inferensial)",
  logreg: "Regresi Logistik (L2)",
  random_forest: "Random Forest",
  hist_gb: "Gradient Boosting",
  isolation_forest: "Isolation Forest",
  m_score: "Beneish M-Score",
};
export function modelLabel(name: string): string {
  return MODEL_LABELS[name] ?? name;
}
