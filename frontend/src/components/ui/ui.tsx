import type { ReactNode } from "react";
import { motion } from "framer-motion";
import { formatInt, formatNumber } from "../../lib/format.ts";
import { useCountUp } from "../../hooks/motion.ts";
import { useStore } from "../../state/store.ts";
import type { Async as AsyncState, DatasetInfo } from "../../state/store.ts";

export function Card(props: { title?: string; sub?: string; action?: ReactNode; children?: ReactNode; className?: string }) {
  return (
    <section className={`card ${props.className ?? ""}`}>
      {(props.title || props.action) && (
        <div className="row between" style={{ marginBottom: props.sub ? 0 : 10 }}>
          <div>
            {props.title && <h3 className="card-title">{props.title}</h3>}
            {props.sub && <p className="card-sub">{props.sub}</p>}
          </div>
          {props.action}
        </div>
      )}
      {props.children}
    </section>
  );
}

export function PageHead(props: { title: string; subtitle?: string }) {
  return (
    <header className="page-head">
      <h1>{props.title}</h1>
      {props.subtitle && <p>{props.subtitle}</p>}
    </header>
  );
}

export function Button(props: {
  children?: ReactNode; onClick?: () => void; variant?: "primary" | "accent" | "ghost" | "light"; size?: "md" | "sm";
  disabled?: boolean; title?: string; type?: "button" | "submit";
}) {
  const cls = `btn btn-${props.variant ?? "primary"}${props.size === "sm" ? " btn-sm" : ""}`;
  return (
    <button type={props.type ?? "button"} className={cls} disabled={props.disabled} title={props.title} onClick={props.onClick}>
      {props.children}
    </button>
  );
}

export function Badge(props: { tone?: "high" | "mid" | "low" | "info" | "error" | "warning" | "ok" | "neutral"; children?: ReactNode }) {
  return <span className={`badge ${props.tone ?? "neutral"}`}>{props.children}</span>;
}

export function Spinner() {
  return <span className="spinner" role="status" aria-label="Memuat" />;
}

export function Loading(props: { text?: string }) {
  return (
    <div className="row" style={{ padding: "26px 4px", color: "var(--muted)" }}>
      <Spinner /> <span>{props.text ?? "Memuat…"}</span>
    </div>
  );
}

export function ErrorBox(props: { message: string; onRetry?: () => void }) {
  return (
    <div className="alert error" role="alert">
      <div className="row between">
        <span>{props.message}</span>
        {props.onRetry && <Button variant="ghost" size="sm" onClick={props.onRetry}>Coba lagi</Button>}
      </div>
    </div>
  );
}

export function Empty(props: { title: string; hint?: string; action?: ReactNode }) {
  return (
    <div className="empty">
      <h3>{props.title}</h3>
      {props.hint && <p>{props.hint}</p>}
      {props.action}
    </div>
  );
}

/** Menampilkan spinner/galat untuk operasi asinkron; anak hanya dirender bila ``ready``. */
export function Async(props: { state: AsyncState; ready: boolean; onRetry?: () => void; loadingText?: string; children: ReactNode }) {
  if (props.state.error) return <ErrorBox message={props.state.error} {...(props.onRetry ? { onRetry: props.onRetry } : {})} />;
  if (!props.ready) return <Loading {...(props.loadingText ? { text: props.loadingText } : {})} />;
  return <>{props.children}</>;
}

export function StatCard(props: { label: string; value: number | null; format?: (n: number) => string; hint?: string }) {
  const shown = useCountUp(props.value);
  const fmt = props.format ?? ((n: number) => formatNumber(n, 0));
  return (
    <motion.div className="card stat" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35 }}>
      <div className="stat-label">{props.label}</div>
      <div className="stat-value num">{shown === null ? "–" : fmt(shown)}</div>
      {props.hint && <div className="stat-hint">{props.hint}</div>}
    </motion.div>
  );
}

export function Tabs<T extends string>(props: { tabs: readonly { id: T; label: string }[]; active: T; onChange: (id: T) => void }) {
  return (
    <div className="tabs" role="tablist">
      {props.tabs.map((t) => (
        <button key={t.id} role="tab" className="tab" aria-selected={t.id === props.active} onClick={() => props.onChange(t.id)}>
          {t.label}
        </button>
      ))}
    </div>
  );
}

/** Pembatas halaman yang butuh dataset: arahkan pengguna ke halaman unggah. */
export function NeedDataset(props: { children: ReactNode }) {
  const dataset = useStore((s) => s.dataset);
  const setPage = useStore((s) => s.setPage);
  const loadSample = useStore((s) => s.loadSample);
  const loading = useStore((s) => s.status.upload.loading);
  if (dataset) return <>{props.children}</>;
  return (
    <Card>
      <Empty
        title="Belum ada dataset"
        hint="Unggah data laporan keuangan, atau gunakan data sampel sintetis untuk mencoba sistem."
        action={
          <div className="row" style={{ justifyContent: "center", marginTop: 12 }}>
            <Button onClick={() => setPage("unggah")}>Unggah data</Button>
            <Button variant="ghost" disabled={loading} onClick={() => void loadSample("bersih")}>
              {loading ? "Memuat…" : "Pakai data sampel"}
            </Button>
          </div>
        }
      />
    </Card>
  );
}

export function KeyValue(props: { items: readonly (readonly [string, ReactNode])[] }) {
  return (
    <dl className="kv">
      {props.items.map(([k, v]) => (
        <div key={k} style={{ display: "contents" }}>
          <dt>{k}</dt>
          <dd>{v}</dd>
        </div>
      ))}
    </dl>
  );
}

/** Ringkasan dataset aktif: jumlah baris, perusahaan, rentang tahun, label, missing, outlier. */
export function DatasetSummary(props: { dataset: DatasetInfo }) {
  const sm = props.dataset.validation.summary;
  const years = sm.rentang_tahun;
  const num = (k: string): number | null => (typeof sm[k] === "number" ? (sm[k] as number) : null);
  return (
    <div className="grid cols-4">
      <div><div className="stat-label">Baris</div><div className="stat-value num">{formatInt(props.dataset.nRows)}</div></div>
      <div><div className="stat-label">Perusahaan</div><div className="stat-value num">{formatInt(sm.n_perusahaan ?? null)}</div></div>
      <div><div className="stat-label">Rentang tahun</div><div className="stat-value num" style={{ fontSize: 24 }}>{years ? `${years[0]}–${years[1]}` : "–"}</div></div>
      <div>
        <div className="stat-label">Label fraud</div>
        <div className="stat-value" style={{ fontSize: 24 }}>{props.dataset.hasLabel ? "Ada" : "Tidak ada"}</div>
        <div className="stat-hint">{num("baris_dengan_missing") ?? 0} baris dengan missing · {num("jumlah_sel_outlier") ?? 0} sel outlier</div>
      </div>
    </div>
  );
}
