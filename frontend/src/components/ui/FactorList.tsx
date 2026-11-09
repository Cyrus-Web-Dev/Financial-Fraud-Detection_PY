import { formatNumber, formatSmart } from "../../lib/format.ts";
import type { Factor } from "../../lib/types.ts";

/** Faktor utama penjelasan: batang merah menaikkan risiko, hijau menurunkan; panjang proporsional terhadap faktor terbesar. */
export function FactorList(props: { factors: readonly Factor[]; unit: string }) {
  const max = Math.max(1e-12, ...props.factors.map((f) => Math.abs(f.kontribusi)));
  const fmt = (v: number) => (props.unit === "probabilitas" ? `${v > 0 ? "+" : "−"}${formatNumber(Math.abs(v) * 100, 1)} poin` : `${v > 0 ? "+" : "−"}${formatNumber(Math.abs(v), 2)} log-odds`);
  return (
    <div style={{ marginTop: 6 }}>
      {props.factors.map((f) => {
        const w = (Math.abs(f.kontribusi) / max) * 50;
        const up = f.kontribusi > 0;
        return (
          <div key={f.fitur} className="factor">
            <div><strong>{f.label}</strong><div className="small muted">nilai {formatSmart(f.nilai)} · {fmt(f.kontribusi)}</div></div>
            <div className="factor-bar" aria-hidden="true"><i style={{ left: up ? "50%" : `${50 - w}%`, width: `${w}%`, background: up ? "var(--red)" : "var(--green)" }} /></div>
          </div>
        );
      })}
    </div>
  );
}
