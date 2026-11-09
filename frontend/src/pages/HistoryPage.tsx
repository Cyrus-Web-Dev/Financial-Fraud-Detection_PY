import { useEffect, useState } from "react";
import { useStore } from "../state/store.ts";
import { Async, Badge, Button, Card, PageHead } from "../components/ui/ui.tsx";
import { DataTable } from "../components/ui/DataTable.tsx";
import { formatInt, formatPercent, formatProb, modelLabel, riskTone } from "../lib/format.ts";

export function HistoryPage() {
  const history = useStore((s) => s.history);
  const status = useStore((s) => s.status.history);
  const fetchHistory = useStore((s) => s.fetchHistory);
  const clear = useStore((s) => s.clearHistory);
  const [model, setModel] = useState("");
  const [flagged, setFlagged] = useState(false);
  const [confirm, setConfirm] = useState(false);

  useEffect(() => {
    void fetchHistory({ ...(model ? { model } : {}), ...(flagged ? { only_flagged: true } : {}) });
  }, [fetchHistory, model, flagged]);

  return (
    <div className="stack">
      <PageHead title="Riwayat Prediksi" subtitle="Setiap prediksi tersimpan di basis data SQLite lokal (100 terbaru ditampilkan)." />
      <Card>
        <div className="row between" style={{ marginBottom: 12 }}>
          <div className="row">
            <select className="select" value={model} onChange={(e) => setModel(e.target.value)} aria-label="Filter model">
              <option value="">Semua model</option>
              {["logit_mle", "logreg", "random_forest", "hist_gb"].map((m) => <option key={m} value={m}>{modelLabel(m)}</option>)}
            </select>
            <label className="toggle"><input type="checkbox" checked={flagged} onChange={(e) => setFlagged(e.target.checked)} /> Hanya yang ditandai</label>
            <span className="small muted">Total tersimpan: {formatInt(history?.total ?? null)}</span>
          </div>
          {confirm ? (
            <div className="row"><span className="small">Hapus semua riwayat?</span><Button variant="accent" size="sm" onClick={() => { void clear(); setConfirm(false); }}>Ya, hapus</Button><Button variant="ghost" size="sm" onClick={() => setConfirm(false)}>Batal</Button></div>
          ) : <Button variant="ghost" size="sm" disabled={!history || history.total === 0} onClick={() => setConfirm(true)}>Hapus riwayat</Button>}
        </div>
        <Async state={status} ready={history !== null} onRetry={() => void fetchHistory()}>
          {history && (
            <DataTable rows={history.data} pageSize={15} empty="Belum ada riwayat. Jalankan prediksi terlebih dahulu." rowKey={(r) => String(r.id)} columns={[
              { key: "t", label: "Waktu", render: (r) => <span className="small">{r.dibuat.replace("T", " ").slice(0, 19)}</span>, sortValue: (r) => r.dibuat },
              { key: "c", label: "Perusahaan", render: (r) => <span className="mono">{r.company_id ?? "–"}</span>, sortValue: (r) => r.company_id },
              { key: "y", label: "Tahun", render: (r) => String(r.tahun ?? "–"), sortValue: (r) => r.tahun },
              { key: "m", label: "Model", render: (r) => modelLabel(r.model) },
              { key: "p", label: "Probabilitas", align: "right", render: (r) => formatProb(r.probabilitas), sortValue: (r) => r.probabilitas },
              { key: "a", label: "Ambang", align: "right", render: (r) => formatPercent(r.ambang, 1) },
              { key: "r", label: "Risiko", render: (r) => <Badge tone={riskTone(r.tingkat_risiko)}>{r.tingkat_risiko}</Badge> },
              { key: "s", label: "Sumber", render: (r) => <span className="small muted">{r.sumber}</span> },
            ]} />
          )}
        </Async>
      </Card>
    </div>
  );
}
