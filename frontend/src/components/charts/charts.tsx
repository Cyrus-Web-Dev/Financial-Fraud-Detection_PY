import type { ReactNode } from "react";
import { divergingColor, extent, linePath, linearScale, niceDomain, percentileColor, ticksIn, fitLabel } from "../../lib/scales.ts";
import { formatNumber, formatSmart } from "../../lib/format.ts";
import type { Bar } from "../../lib/chartData.ts";

export const PALETTE = ["#12305a", "#d7263d", "#2f6fde", "#e08a14", "#1a9a68", "#7a5cc4", "#5b6b85"];
const W = 640;

function Frame(props: { height: number; children: ReactNode; title?: string }) {
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${props.height}`} role="img" aria-label={props.title ?? "grafik"}>
      {props.title && <title>{props.title}</title>}
      {props.children}
    </svg>
  );
}

export function Legend(props: { items: readonly { label: string; color: string }[] }) {
  return (
    <div className="legend">
      {props.items.map((i) => (
        <span key={i.label}><i style={{ background: i.color }} />{i.label}</span>
      ))}
    </div>
  );
}

type Fmt = (n: number) => string;

const fit = fitLabel;

/** Bar vertikal dengan error bar opsional (lo/hi) dan label nilai. */
export function BarChart(props: { data: readonly Bar[]; height?: number; format?: Fmt; yDomain?: [number, number]; color?: string; title?: string; yTitle?: string }) {
  const H = props.height ?? 280;
  const m = { l: 56, r: 12, t: 16, b: props.data.some((d) => d.label.length > 9) ? 78 : 40 };
  const fmt = props.format ?? ((n: number) => formatSmart(n));
  const top = Math.max(0, ...props.data.map((d) => d.hi ?? d.value));
  const [d0, d1] = props.yDomain ?? niceDomain(0, top * 1.08 || 1);
  const y = linearScale(d0, d1, H - m.b, m.t);
  const band = (W - m.l - m.r) / Math.max(1, props.data.length);
  const bw = Math.min(56, band * 0.6);
  const rotate = props.data.some((d) => d.label.length > 9);
  return (
    <Frame height={H} {...(props.title ? { title: props.title } : {})}>
      {ticksIn(d0, d1, 5).map((t) => (
        <g key={t}>
          <line className="grid-line" x1={m.l} x2={W - m.r} y1={y(t)} y2={y(t)} />
          <text x={m.l - 8} y={y(t) + 4} textAnchor="end">{fmt(t)}</text>
        </g>
      ))}
      {props.yTitle && <text x={4} y={12} className="title-text">{props.yTitle}</text>}
      {props.data.map((d, i) => {
        const cx = m.l + band * (i + 0.5);
        const x0 = cx - bw / 2;
        return (
          <g key={`${d.label}-${i}`}>
            <rect x={x0} y={y(Math.max(d.value, 0))} width={bw} height={Math.max(1, Math.abs(y(d.value) - y(0)))} rx={4} fill={props.color ?? PALETTE[2]} opacity={0.9} />
            {d.lo !== undefined && d.hi !== undefined && (
              <g stroke="#0f1b33" strokeWidth={1.4}>
                <line x1={cx} x2={cx} y1={y(d.lo)} y2={y(d.hi)} />
                <line x1={cx - 5} x2={cx + 5} y1={y(d.lo)} y2={y(d.lo)} />
                <line x1={cx - 5} x2={cx + 5} y1={y(d.hi)} y2={y(d.hi)} />
              </g>
            )}
            <text x={cx} y={y(d.hi ?? d.value) - 6} textAnchor="middle" className="title-text">{fmt(d.value)}</text>
            <text x={cx} y={H - m.b + 16} textAnchor={rotate ? "end" : "middle"} transform={rotate ? `rotate(-32 ${cx} ${H - m.b + 16})` : undefined}>{d.label}</text>
          </g>
        );
      })}
      <line className="axis-line" x1={m.l} x2={W - m.r} y1={y(0)} y2={y(0)} />
    </Frame>
  );
}

/** Bar horizontal (nilai dapat bertanda). Label panjang di sisi kiri. */
export function HBarChart(props: { data: readonly Bar[]; format?: Fmt; colorFn?: (b: Bar) => string; labelWidth?: number; title?: string; rowHeight?: number }) {
  const rh = props.rowHeight ?? 30;
  const lw = props.labelWidth ?? 210;
  const H = props.data.length * rh + 36;
  const fmt = props.format ?? ((n: number) => formatSmart(n));
  const ext = extent(props.data.flatMap((d) => [d.value, d.lo, d.hi])) ?? [0, 1];
  const [d0, d1] = niceDomain(Math.min(0, ext[0]), Math.max(0, ext[1]) * 1.12 || 1);
  const x = linearScale(d0, d1, lw, W - 56);
  return (
    <Frame height={H} {...(props.title ? { title: props.title } : {})}>
      {ticksIn(d0, d1, 5).map((t) => (
        <g key={t}>
          <line className="grid-line" x1={x(t)} x2={x(t)} y1={4} y2={H - 26} />
          <text x={x(t)} y={H - 8} textAnchor="middle">{fmt(t)}</text>
        </g>
      ))}
      {props.data.map((d, i) => {
        const cy = 12 + i * rh + rh / 2 - 4;
        const x0 = x(Math.min(0, d.value));
        const w = Math.abs(x(d.value) - x(0));
        return (
          <g key={`${d.label}-${i}`}>
            <text x={lw - 10} y={cy + 4} textAnchor="end">{fit(d.label, lw - 16)}</text>
            <rect x={x0} y={cy - 9} width={Math.max(1, w)} height={18} rx={4} fill={props.colorFn ? props.colorFn(d) : PALETTE[2]} opacity={0.92} />
            <text x={d.value >= 0 ? x(d.value) + 6 : x(d.value) - 6} y={cy + 4} textAnchor={d.value >= 0 ? "start" : "end"} className="title-text">{fmt(d.value)}</text>
          </g>
        );
      })}
      <line className="axis-line" x1={x(0)} x2={x(0)} y1={4} y2={H - 26} />
    </Frame>
  );
}

export type LineSeries = { name: string; points: readonly (readonly [number, number])[]; color?: string; dashed?: boolean; label?: string };

/** Grafik garis multi-seri; ``diagonal`` menggambar garis acuan y = x (ROC/kalibrasi). */
export function LineChart(props: {
  series: readonly LineSeries[]; xLabel: string; yLabel: string; xDomain?: [number, number]; yDomain?: [number, number]; height?: number;
  diagonal?: boolean; xFormat?: Fmt; yFormat?: Fmt; title?: string; showLegend?: boolean; xTicks?: readonly number[];
}) {
  const H = props.height ?? 320;
  const m = { l: 58, r: 16, t: 14, b: 46 };
  const all = props.series.flatMap((s) => s.points);
  const ex = extent(all.map((p) => p[0])) ?? [0, 1];
  const ey = extent(all.map((p) => p[1])) ?? [0, 1];
  const [x0, x1] = props.xDomain ?? niceDomain(ex[0], ex[1]);
  const [y0, y1] = props.yDomain ?? niceDomain(Math.min(0, ey[0]), ey[1]);
  const x = linearScale(x0, x1, m.l, W - m.r);
  const y = linearScale(y0, y1, H - m.b, m.t);
  const xf = props.xFormat ?? ((n: number) => formatSmart(n));
  const yf = props.yFormat ?? ((n: number) => formatSmart(n));
  const colored = props.series.map((s, i) => ({ ...s, color: s.color ?? PALETTE[i % PALETTE.length] ?? "#12305a" }));
  return (
    <div>
      <Frame height={H} {...(props.title ? { title: props.title } : {})}>
        {ticksIn(y0, y1, 6).map((t) => (
          <g key={`y${t}`}>
            <line className="grid-line" x1={m.l} x2={W - m.r} y1={y(t)} y2={y(t)} />
            <text x={m.l - 8} y={y(t) + 4} textAnchor="end">{yf(t)}</text>
          </g>
        ))}
        {(props.xTicks ?? ticksIn(x0, x1, 6)).map((t) => (
          <g key={`x${t}`}>
            <line className="grid-line" x1={x(t)} x2={x(t)} y1={m.t} y2={H - m.b} />
            <text x={x(t)} y={H - m.b + 16} textAnchor="middle">{xf(t)}</text>
          </g>
        ))}
        <line className="axis-line" x1={m.l} x2={W - m.r} y1={H - m.b} y2={H - m.b} />
        <line className="axis-line" x1={m.l} x2={m.l} y1={m.t} y2={H - m.b} />
        {props.diagonal && <line x1={x(x0)} y1={y(y0)} x2={x(x1)} y2={y(y1)} stroke="#9aa8c0" strokeDasharray="5 5" />}
        {colored.map((s) => (
          <path key={s.name} d={linePath(s.points.map((p) => [x(p[0]), y(p[1])] as const))} fill="none" stroke={s.color} strokeWidth={2.2} strokeLinejoin="round" strokeLinecap="round" strokeDasharray={s.dashed ? "6 5" : undefined} />
        ))}
        <text x={(m.l + W - m.r) / 2} y={H - 6} textAnchor="middle" className="title-text">{props.xLabel}</text>
        <text transform={`rotate(-90 14 ${(m.t + H - m.b) / 2})`} x={14} y={(m.t + H - m.b) / 2} textAnchor="middle" className="title-text">{props.yLabel}</text>
      </Frame>
      {props.showLegend !== false && props.series.length > 1 && <Legend items={colored.map((s) => ({ label: s.label ?? s.name, color: s.color }))} />}
    </div>
  );
}

/** Peta panas matriks persegi (korelasi, kesepakatan peringkat). */
export function Heatmap(props: { labels: readonly string[]; values: readonly (readonly (number | null)[])[]; vmin?: number; vmax?: number; format?: Fmt; title?: string }) {
  const n = props.labels.length;
  const left = 120;
  const top = 110;
  const cell = Math.max(18, Math.min(52, (W - left - 96) / Math.max(1, n)));
  const H = top + cell * n + 10;
  const fmt = props.format ?? ((v: number) => formatNumber(v, 2));
  return (
    <Frame height={H} {...(props.title ? { title: props.title } : {})}>
      {props.labels.map((l, i) => (
        <g key={l}>
          <text x={left - 8} y={top + cell * i + cell / 2 + 4} textAnchor="end">{fit(l, left - 14)}</text>
          <text transform={`rotate(-45 ${left + cell * i + cell / 2} ${top - 8})`} x={left + cell * i + cell / 2} y={top - 8} textAnchor="start">{fit(l, 90)}</text>
        </g>
      ))}
      {props.values.map((row, i) =>
        row.map((v, j) => (
          <g key={`${i}-${j}`}>
            <rect x={left + cell * j} y={top + cell * i} width={cell - 1} height={cell - 1} rx={3} fill={divergingColor(v, props.vmin ?? -1, props.vmax ?? 1)}>
              <title>{`${props.labels[i] ?? ""} × ${props.labels[j] ?? ""}: ${v === null ? "–" : fmt(v)}`}</title>
            </rect>
            {cell >= 34 && v !== null && (
              <text x={left + cell * j + cell / 2} y={top + cell * i + cell / 2 + 4} textAnchor="middle" style={{ fill: Math.abs(v) > 0.6 ? "#fff" : "#0f1b33", fontSize: 10.5 }}>{fmt(v)}</text>
            )}
          </g>
        )),
      )}
    </Frame>
  );
}

/** Scatter plot; ``t`` (0-1) mewarnai titik dengan skala persentil biru-merah. */
export function ScatterChart(props: { points: readonly { x: number; y: number; t?: number }[]; xLabel: string; yLabel: string; height?: number; zeroLine?: boolean; title?: string }) {
  const H = props.height ?? 300;
  const m = { l: 58, r: 16, t: 14, b: 46 };
  const ex = extent(props.points.map((p) => p.x)) ?? [0, 1];
  const ey = extent(props.points.map((p) => p.y)) ?? [0, 1];
  const [x0, x1] = niceDomain(ex[0], ex[1]);
  const [y0, y1] = niceDomain(Math.min(ey[0], props.zeroLine ? 0 : ey[0]), Math.max(ey[1], props.zeroLine ? 0 : ey[1]));
  const x = linearScale(x0, x1, m.l, W - m.r);
  const y = linearScale(y0, y1, H - m.b, m.t);
  return (
    <Frame height={H} {...(props.title ? { title: props.title } : {})}>
      {ticksIn(y0, y1, 5).map((t) => (
        <g key={`y${t}`}><line className="grid-line" x1={m.l} x2={W - m.r} y1={y(t)} y2={y(t)} /><text x={m.l - 8} y={y(t) + 4} textAnchor="end">{formatSmart(t)}</text></g>
      ))}
      {ticksIn(x0, x1, 6).map((t) => (
        <g key={`x${t}`}><line className="grid-line" x1={x(t)} x2={x(t)} y1={m.t} y2={H - m.b} /><text x={x(t)} y={H - m.b + 16} textAnchor="middle">{formatSmart(t)}</text></g>
      ))}
      {props.zeroLine && <line x1={m.l} x2={W - m.r} y1={y(0)} y2={y(0)} stroke="#9aa8c0" strokeDasharray="4 4" />}
      {props.points.map((p, i) => (
        <circle key={i} cx={x(p.x)} cy={y(p.y)} r={3.2} fill={p.t === undefined ? PALETTE[2] : percentileColor(p.t)} opacity={0.75} />
      ))}
      <text x={(m.l + W - m.r) / 2} y={H - 6} textAnchor="middle" className="title-text">{props.xLabel}</text>
      <text transform={`rotate(-90 14 ${(m.t + H - m.b) / 2})`} x={14} y={(m.t + H - m.b) / 2} textAnchor="middle" className="title-text">{props.yLabel}</text>
    </Frame>
  );
}

/** Forest plot odds ratio pada skala log; garis acuan di OR = 1. */
export function ForestPlot(props: { rows: readonly { label: string; or: number; lo: number; hi: number; sig: boolean }[]; title?: string }) {
  const rh = 32;
  const lw = 230;
  const H = props.rows.length * rh + 44;
  const lows = props.rows.map((r) => r.lo).filter((v) => v > 0);
  const highs = props.rows.map((r) => r.hi).filter((v) => Number.isFinite(v));
  const lo = Math.min(0.5, ...lows) * 0.9;
  const hi = Math.max(2, ...highs) * 1.1;
  const x = linearScale(Math.log(lo), Math.log(hi), lw, W - 24);
  const ticks = [0.25, 0.5, 1, 2, 4, 8, 16].filter((t) => t >= lo && t <= hi);
  return (
    <Frame height={H} {...(props.title ? { title: props.title } : {})}>
      {ticks.map((t) => (
        <g key={t}><line className="grid-line" x1={x(Math.log(t))} x2={x(Math.log(t))} y1={4} y2={H - 28} /><text x={x(Math.log(t))} y={H - 10} textAnchor="middle">{formatNumber(t, t < 1 ? 2 : 0)}</text></g>
      ))}
      <line x1={x(0)} x2={x(0)} y1={4} y2={H - 28} stroke="#d7263d" strokeDasharray="4 4" />
      {props.rows.map((r, i) => {
        const cy = 14 + i * rh + rh / 2 - 4;
        const col = r.sig ? "#12305a" : "#9aa8c0";
        return (
          <g key={r.label}>
            <text x={lw - 10} y={cy + 4} textAnchor="end">{fit(r.label, lw - 16)}</text>
            <line x1={x(Math.log(Math.max(r.lo, lo)))} x2={x(Math.log(Math.min(r.hi, hi)))} y1={cy} y2={cy} stroke={col} strokeWidth={2.2} />
            <rect x={x(Math.log(r.or)) - 5} y={cy - 5} width={10} height={10} rx={2} fill={col} transform={`rotate(45 ${x(Math.log(r.or))} ${cy})`} />
          </g>
        );
      })}
    </Frame>
  );
}

/** Summary plot SHAP: satu baris per fitur, titik diwarnai persentil nilai fitur. */
export function Beeswarm(props: { groups: readonly { label: string; points: readonly { shap: number; t: number }[] }[]; xLabel: string; title?: string }) {
  const rh = 44;
  const lw = 210;
  const H = props.groups.length * rh + 56;
  const lim = Math.max(1e-9, ...props.groups.flatMap((g) => g.points.map((p) => Math.abs(p.shap))));
  const [d0, d1] = niceDomain(-lim, lim);
  const x = linearScale(d0, d1, lw, W - 20);
  return (
    <div>
      <Frame height={H} {...(props.title ? { title: props.title } : {})}>
        {ticksIn(d0, d1, 6).map((t) => (
          <g key={t}><line className="grid-line" x1={x(t)} x2={x(t)} y1={4} y2={H - 42} /><text x={x(t)} y={H - 26} textAnchor="middle">{formatSmart(t)}</text></g>
        ))}
        <line x1={x(0)} x2={x(0)} y1={4} y2={H - 42} stroke="#9aa8c0" />
        {props.groups.map((g, gi) => {
          const cy = 8 + gi * rh + rh / 2;
          return (
            <g key={g.label}>
              <text x={lw - 10} y={cy + 4} textAnchor="end">{fit(g.label, lw - 16)}</text>
              {g.points.map((p, i) => (
                <circle key={i} cx={x(p.shap)} cy={cy + (((i * 0.6180339) % 1) - 0.5) * rh * 0.75} r={2.8} fill={percentileColor(p.t)} opacity={0.72} />
              ))}
            </g>
          );
        })}
        <text x={(lw + W - 20) / 2} y={H - 8} textAnchor="middle" className="title-text">{props.xLabel}</text>
      </Frame>
      <div className="legend"><span><i style={{ background: percentileColor(0) }} />Nilai fitur rendah</span><span><i style={{ background: percentileColor(1) }} />Nilai fitur tinggi</span></div>
    </div>
  );
}

/** Diagram reliabilitas: prediksi vs frekuensi teramati dengan CI 95% dan garis ideal. */
export function ReliabilityChart(props: { points: readonly { x: number; y: number; lo: number; hi: number; n: number }[]; title?: string }) {
  const H = 340;
  const m = { l: 58, r: 16, t: 14, b: 46 };
  const top = Math.max(0.05, ...props.points.flatMap((p) => [p.x, p.hi])) * 1.08;
  const [, d1] = niceDomain(0, top);
  const x = linearScale(0, d1, m.l, W - m.r);
  const y = linearScale(0, d1, H - m.b, m.t);
  return (
    <Frame height={H} {...(props.title ? { title: props.title } : {})}>
      {ticksIn(0, d1, 6).map((t) => (
        <g key={t}>
          <line className="grid-line" x1={m.l} x2={W - m.r} y1={y(t)} y2={y(t)} />
          <line className="grid-line" x1={x(t)} x2={x(t)} y1={m.t} y2={H - m.b} />
          <text x={m.l - 8} y={y(t) + 4} textAnchor="end">{formatNumber(t * 100, 0)}%</text>
          <text x={x(t)} y={H - m.b + 16} textAnchor="middle">{formatNumber(t * 100, 0)}%</text>
        </g>
      ))}
      <line x1={x(0)} y1={y(0)} x2={x(d1)} y2={y(d1)} stroke="#9aa8c0" strokeDasharray="5 5" />
      <path d={linePath(props.points.map((p) => [x(p.x), y(p.y)] as const))} fill="none" stroke="#12305a" strokeWidth={2} />
      {props.points.map((p, i) => (
        <g key={i}>
          <line x1={x(p.x)} x2={x(p.x)} y1={y(p.lo)} y2={y(p.hi)} stroke="#d7263d" strokeWidth={1.6} />
          <circle cx={x(p.x)} cy={y(p.y)} r={4.5} fill="#12305a"><title>{`n = ${p.n}`}</title></circle>
        </g>
      ))}
      <text x={(m.l + W - m.r) / 2} y={H - 6} textAnchor="middle" className="title-text">Probabilitas prediksi (rata-rata per bin)</text>
      <text transform={`rotate(-90 14 ${(m.t + H - m.b) / 2})`} x={14} y={(m.t + H - m.b) / 2} textAnchor="middle" className="title-text">Frekuensi fraud teramati</text>
    </Frame>
  );
}
