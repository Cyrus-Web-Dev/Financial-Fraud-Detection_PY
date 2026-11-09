import { useEffect, useMemo, useState } from "react";
import { useStore } from "../state/store.ts";
import { Async, Badge, Button, Card, NeedDataset, PageHead } from "../components/ui/ui.tsx";
import { DataTable } from "../components/ui/DataTable.tsx";
import type { Column } from "../components/ui/DataTable.tsx";
import { BarChart } from "../components/charts/charts.tsx";
import { RiskGauge3D } from "../components/three/RiskGauge3D.tsx";
import { ErrorBoundary } from "../components/layout/ErrorBoundary.tsx";
import { FactorList } from "../components/ui/FactorList.tsx";
import { filterPredictions, paginate, probabilityHistogram, reclassify, summarizePredictions, thresholdMarks } from "../state/logic.ts";
import type { PredictionFilter } from "../state/logic.ts";
import { formatInt, formatNumber, formatPercent, formatProb, modelLabel, riskTone } from "../lib/format.ts";
import type { PredictionRow, RiskLevel } from "../lib/types.ts";

export function PredictPage() {
  const dataset = useStore((s) => s.dataset);
  const models = useStore((s) => s.models);
  const model = useStore((s) => s.model);
  const threshold = useStore((s) => s.threshold);
  const tahun = useStore((s) => s.tahun);
  const explain = useStore((s) => s.explain);
  const prediction = useStore((s) => s.prediction);
  const status = useStore((s) => s.status.predict);
  const setModel = useStore((s) => s.setModel);
  const setThreshold = useStore((s) => s.setThreshold);
  const setTahun = useStore((s) => s.setTahun);
  const setExplain = useStore((s) => s.setExplain);
  const run = useStore((s) => s.runPrediction);
  const setPage = useStore((s) => s.setPage);

  const defaultThr = models?.ambang_default ?? 0.0909;
  const effective = threshold ?? defaultThr;
  const years = useMemo(() => {
    const r = dataset?.validation.summary.rentang_tahun;
    return r ? Array.from({ length: r[1] - r[0] + 1 }, (_, i) => r[0] + i) : [];
  }, [dataset]);

  const [filter, setFilter] = useState<PredictionFilter>({ query: "", onlyFlagged: false, risk: "semua" });
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  useEffect(() => setSelectedKey(null), [prediction]);

  const rows = useMemo(() => (prediction ? reclassify(prediction.hasil, effective) : []), [prediction, effective]);
  const summary = useMemo(() => summarizePredictions(rows), [rows]);
  const filtered = useMemo(() => filterPredictions(rows, filter), [rows, filter]);
  const keyOf = (r: PredictionRow) => `${r.company_id}|${r.tahun}`;
  const top = useMemo(() => [...rows].sort((a, b) => b.probabilitas - a.probabilitas)[0], [rows]);
  const selected = rows.find((r) => keyOf(r) === selectedKey) ?? top;
  const allowedExplain = prediction ? prediction.hasil.length <= 50 : true;

  const cols: Column<PredictionRow>[] = [
    { key: "id", label: "Perusahaan", render: (r) => <span className="mono">{r.company_id}</span>, sortValue: (r) => r.company_id },
    { key: "tahun", label: "Tahun", render: (r) => String(r.tahun), sortValue: (r) => r.tahun },
    { key: "sektor", label: "Sektor", render: (r) => r.sektor, sortValue: (r) => r.sektor },
    { key: "p", label: "Probabilitas", align: "right", render: (r) => formatProb(r.probabilitas), sortValue: (r) => r.probabilitas },
    { key: "risk", label: "Risiko", render: (r) => <Badge tone={riskTone(r.tingkat_risiko)}>{r.tingkat_risiko}</Badge>, sortValue: (r) => r.probabilitas },
    { key: "m", label: "M-Score", align: "right", render: (r) => formatNumber(r.m_score, 2), sortValue: (r) => r.m_score },
    { key: "b", label: "Beneish", render: (r) => (r.m_flag ? <Badge tone="high">tandai</Badge> : <span className="muted small">–</span>) },
  ];

  return (
    <div className="stack">
      <PageHead title="Prediksi Fraud" subtitle="Probabilitas yang ditampilkan sudah dikalibrasi. Geser ambang untuk melihat dampaknya langsung pada jumlah perusahaan yang ditandai." />
      <NeedDataset>
        <div className="grid cols-2" style={{ gridTemplateColumns: "minmax(0, 1.25fr) minmax(0, 1fr)" }}>
          <Card title="1. Pilih model">
            <div className="grid cols-2">
              {(models?.model ?? []).map((m) => (
                <button key={m.nama} className="model-card" aria-pressed={m.nama === model} disabled={!m.tersedia} onClick={() => setModel(m.nama)}>
                  <strong>{m.label}</strong>
                  <div className="small muted">{m.jenis === "klasik" ? "Model klasik, mudah ditafsirkan" : "Machine learning, menangkap nonlinearitas"}</div>
                </button>
              ))}
            </div>
            {!models && <div className="muted small">Memuat daftar model…</div>}
          </Card>
          <Card title="2. Atur ambang dan data">
            <div className="field">
              <label htmlFor="thr">Ambang keputusan: <strong className="num">{formatPercent(effective, 1)}</strong> {threshold === null && <Badge tone="info">default</Badge>}</label>
              <input id="thr" className="range" type="range" min={0.01} max={0.6} step={0.005} value={effective} onChange={(e) => setThreshold(Number(e.target.value))} />
              <div className="row" style={{ gap: 6 }}>
                {thresholdMarks(defaultThr).map((m) => <Button key={m.label + m.value} variant="ghost" size="sm" onClick={() => setThreshold(m.value === defaultThr ? null : m.value)}>{m.label === "default" ? `Default ${formatPercent(m.value, 1)}` : m.label}</Button>)}
              </div>
              <p className="small muted">Ambang default = ambang Bayes dari rasio biaya FN : FP = 10 : 1.</p>
            </div>
            <div className="row" style={{ marginTop: 10 }}>
              <div className="field" style={{ minWidth: 150 }}>
                <label htmlFor="thn">Tahun</label>
                <select id="thn" className="select" value={tahun ?? ""} onChange={(e) => setTahun(e.target.value === "" ? null : Number(e.target.value))}>
                  <option value="">Semua tahun</option>
                  {years.map((y) => <option key={y} value={y}>{y}</option>)}
                </select>
              </div>
              <label className="toggle" title="Penjelasan Shapley hanya dihitung bila jumlah baris ≤ 50">
                <input type="checkbox" checked={explain} onChange={(e) => setExplain(e.target.checked)} /> Sertakan penjelasan (≤ 50 baris)
              </label>
            </div>
            <div className="row" style={{ marginTop: 14 }}>
              <Button variant="accent" disabled={status.loading || !models} onClick={() => void run()}>{status.loading ? "Menghitung…" : "Jalankan prediksi"}</Button>
              {prediction && <span className="small muted">{prediction.label_model} · {formatInt(prediction.n_baris)} baris</span>}
            </div>
          </Card>
        </div>

        <Async state={status} ready={!status.loading} loadingText="Menghitung prediksi…">
          {prediction && (
            <div className="stack">
              {prediction.catatan.map((c) => <div key={c} className="alert info">{c}</div>)}
              <div className="grid cols-4">
                <Card><div className="stat-label">Ditandai</div><div className="stat-value num">{formatInt(summary.flagged)}</div><div className="stat-hint">{formatPercent(summary.total ? summary.flagged / summary.total : 0, 1)} dari {formatInt(summary.total)}</div></Card>
                <Card><div className="stat-label">Risiko tinggi</div><div className="stat-value num" style={{ color: "var(--red)" }}>{formatInt(summary.tinggi)}</div></Card>
                <Card><div className="stat-label">Risiko sedang</div><div className="stat-value num" style={{ color: "var(--amber)" }}>{formatInt(summary.sedang)}</div></Card>
                <Card><div className="stat-label">Rata-rata probabilitas</div><div className="stat-value num">{formatPercent(summary.meanProb, 1)}</div><div className="stat-hint">maksimum {formatPercent(summary.maxProb, 1)}</div></Card>
              </div>
              <div className="grid cols-2" style={{ gridTemplateColumns: "minmax(0, 1fr) minmax(0, 1fr)" }}>
                <Card title="Sebaran probabilitas" sub="Jumlah perusahaan per rentang probabilitas terkalibrasi."><BarChart data={probabilityHistogram(rows)} height={240} color="#2f6fde" format={(n) => formatInt(n)} /></Card>
                <Card title={selected ? `${selected.company_id} · ${selected.tahun}` : "Perusahaan terpilih"} sub={selected ? `${selected.sektor} · klik baris tabel untuk mengganti` : undefined}>
                  {selected && (
                    <>
                      <ErrorBoundary fallback={<div className="stat-value">{formatProb(selected.probabilitas)}</div>}>
                        <RiskGauge3D probability={selected.probabilitas} tone={riskTone(selected.tingkat_risiko)} caption={selected.tingkat_risiko} />
                      </ErrorBoundary>
                      {selected.faktor_utama ? <FactorList factors={selected.faktor_utama} unit={prediction.satuan_penjelasan} /> : (
                        <p className="small muted" style={{ textAlign: "center" }}>{allowedExplain ? "Aktifkan penjelasan lalu jalankan ulang untuk melihat faktor utama." : "Penjelasan per baris hanya untuk ≤ 50 baris. Pilih satu tahun/perusahaan di halaman Analisis Perusahaan."}</p>
                      )}
                      <div className="row" style={{ justifyContent: "center", marginTop: 8 }}><Button variant="ghost" size="sm" onClick={() => setPage("perusahaan")}>Analisis perusahaan →</Button></div>
                    </>
                  )}
                </Card>
              </div>
              <Card title="Hasil per perusahaan" sub={`${modelLabel(prediction.model)} · ambang ${formatPercent(effective, 1)}`}>
                <div className="row" style={{ marginBottom: 12 }}>
                  <input className="input" placeholder="Cari perusahaan, sektor, tahun…" value={filter.query} onChange={(e) => setFilter({ ...filter, query: e.target.value })} aria-label="Cari" />
                  <select className="select" value={filter.risk} onChange={(e) => setFilter({ ...filter, risk: e.target.value as RiskLevel | "semua" })} aria-label="Filter risiko">
                    <option value="semua">Semua risiko</option><option value="Tinggi">Tinggi</option><option value="Sedang">Sedang</option><option value="Rendah">Rendah</option>
                  </select>
                  <label className="toggle"><input type="checkbox" checked={filter.onlyFlagged} onChange={(e) => setFilter({ ...filter, onlyFlagged: e.target.checked })} /> Hanya yang ditandai</label>
                  <span className="small muted">{formatInt(paginate(filtered, 1, 1).total)} baris</span>
                </div>
                <DataTable rows={filtered} columns={cols} pageSize={12} initialSort={{ key: "p", dir: "desc" }} rowKey={keyOf} onRowClick={(r) => setSelectedKey(keyOf(r))} isSelected={(r) => keyOf(r) === (selected ? keyOf(selected) : "")} empty="Tidak ada baris yang cocok dengan filter." />
              </Card>
            </div>
          )}
        </Async>
      </NeedDataset>
    </div>
  );
}
