import { useState } from "react";
import { useStore } from "../state/store.ts";
import { Badge, Button, Card, ErrorBox, KeyValue, NeedDataset, PageHead } from "../components/ui/ui.tsx";
import { formatInt, formatPercent, modelLabel } from "../lib/format.ts";

const SECTIONS = [
  "Ringkasan eksekutif dan lima observasi berisiko tertinggi",
  "Data dan validasi (temuan missing value, outlier, identitas neraca)",
  "Eksplorasi data: proporsi fraud, uji Benford, fraud vs non-fraud",
  "Hasil prediksi: sebaran probabilitas, ringkasan per sektor, daftar teratas dengan faktor utama",
  "Kinerja model: metrik dengan CI 95%, ROC/PR, kalibrasi, ambang berbasis biaya, perbandingan model, diagnostik",
  "Penjelasan: odds ratio dan SHAP eksak",
  "Asumsi, keterbatasan, potensi bias, dan reproducibility",
] as const;

export function ReportPage() {
  const dataset = useStore((s) => s.dataset);
  const models = useStore((s) => s.models);
  const model = useStore((s) => s.model);
  const threshold = useStore((s) => s.threshold);
  const tahun = useStore((s) => s.tahun);
  const setModel = useStore((s) => s.setModel);
  const setTahun = useStore((s) => s.setTahun);
  const report = useStore((s) => s.report);
  const status = useStore((s) => s.status.report);
  const generate = useStore((s) => s.generateReport);
  const [topN, setTopN] = useState(20);

  const years = (() => {
    const r = dataset?.validation.summary.rentang_tahun;
    return r ? Array.from({ length: r[1] - r[0] + 1 }, (_, i) => r[0] + i) : [];
  })();
  const effective = threshold ?? models?.ambang_default ?? 0.0909;

  return (
    <div className="stack">
      <PageHead title="Laporan PDF" subtitle="Ekspor ringkasan analisis (data, prediksi, kinerja, penjelasan, dan keterbatasan) menjadi dokumen PDF yang siap dilampirkan pada skripsi atau laporan audit." />
      <NeedDataset>
        <div className="grid cols-2" style={{ gridTemplateColumns: "minmax(0, 1fr) minmax(0, 1fr)" }}>
          <Card title="Pengaturan laporan">
            <div className="field">
              <label htmlFor="rm">Model prediksi</label>
              <select id="rm" className="select" value={model} onChange={(e) => setModel(e.target.value as typeof model)}>
                {(models?.model ?? []).filter((m) => m.tersedia).map((m) => <option key={m.nama} value={m.nama}>{m.label}</option>)}
              </select>
            </div>
            <div className="row" style={{ marginTop: 10 }}>
              <div className="field" style={{ minWidth: 160 }}>
                <label htmlFor="rt">Tahun</label>
                <select id="rt" className="select" value={tahun ?? ""} onChange={(e) => setTahun(e.target.value === "" ? null : Number(e.target.value))}>
                  <option value="">Semua tahun</option>
                  {years.map((y) => <option key={y} value={y}>{y}</option>)}
                </select>
              </div>
              <div className="field" style={{ minWidth: 160 }}>
                <label htmlFor="rn">Observasi teratas</label>
                <select id="rn" className="select" value={topN} onChange={(e) => setTopN(Number(e.target.value))}>
                  {[10, 20, 30, 50].map((n) => <option key={n} value={n}>{n} teratas (dengan faktor utama)</option>)}
                </select>
              </div>
            </div>
            <p className="small muted" style={{ marginTop: 10 }}>Ambang keputusan mengikuti pengaturan halaman Prediksi: <strong>{formatPercent(effective, 1)}</strong>{threshold === null ? " (default)" : ""}.</p>
            <div className="row" style={{ marginTop: 12 }}>
              <Button variant="accent" disabled={status.loading || !models} onClick={() => void generate(topN)}>{status.loading ? "Membuat laporan…" : "Buat laporan PDF"}</Button>
              {status.loading && <span className="small muted">Menyusun grafik dan tabel (beberapa detik)…</span>}
            </div>
            {status.error && <div style={{ marginTop: 12 }}><ErrorBox message={status.error} /></div>}
          </Card>
          <Card title="Isi laporan">
            <ol className="small" style={{ paddingLeft: 18, margin: 0 }}>{SECTIONS.map((s) => <li key={s} style={{ marginBottom: 4 }}>{s}</li>)}</ol>
            <p className="small muted" style={{ marginTop: 10 }}>Bagian yang datanya belum tersedia dilewati otomatis. Penomoran bagian menyesuaikan.</p>
          </Card>
        </div>

        {report && (
          <Card title="Laporan siap" action={<Badge tone="ok">PDF · {formatInt(Math.round(report.size / 1024))} KB</Badge>}>
            <KeyValue items={[["Nama berkas", <span className="mono">{report.filename}</span>], ["Model", modelLabel(report.model)], ["Cakupan", report.tahun === null ? "Semua tahun" : `Tahun ${report.tahun}`]]} />
            <div className="row" style={{ marginTop: 14 }}>
              <a className="btn btn-primary" href={report.url} download={report.filename}>Unduh PDF</a>
              <a className="btn btn-ghost" href={report.url} target="_blank" rel="noopener noreferrer">Buka di tab baru</a>
            </div>
            <iframe title="Pratinjau laporan PDF" src={report.url} style={{ width: "100%", height: 640, border: "1px solid var(--line)", borderRadius: 12, marginTop: 16, background: "#fff" }} />
          </Card>
        )}
      </NeedDataset>
    </div>
  );
}
