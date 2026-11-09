import { useRef, useState } from "react";
import type { DragEvent, ChangeEvent } from "react";
import { useStore } from "../state/store.ts";
import { Badge, Button, Card, DatasetSummary, ErrorBox, Loading, PageHead } from "../components/ui/ui.tsx";
import { DataTable } from "../components/ui/DataTable.tsx";
import type { Column } from "../components/ui/DataTable.tsx";
import { groupIssues } from "../state/logic.ts";
import { formatInt, formatPercent } from "../lib/format.ts";
import type { Issue, Row } from "../lib/types.ts";

const MAX_MB = 20;
const OK_EXT = [".csv", ".xlsx"];

function IssueList(props: { issues: Issue[]; tone: "error" | "warning" | "info"; limit: number }) {
  const [all, setAll] = useState(false);
  const shown = all ? props.issues : props.issues.slice(0, props.limit);
  if (props.issues.length === 0) return null;
  return (
    <div style={{ marginTop: 10 }}>
      {shown.map((i, k) => (
        <div key={`${i.code}-${i.column ?? ""}-${k}`} className={`alert ${props.tone}`} style={{ marginBottom: 6 }}>
          <Badge tone={props.tone}>{i.code}</Badge> {i.message}
          {i.share !== null && <span className="muted"> ({formatPercent(i.share, 1)})</span>}
        </div>
      ))}
      {props.issues.length > props.limit && (
        <Button variant="ghost" size="sm" onClick={() => setAll(!all)}>{all ? "Ringkas" : `Tampilkan ${props.issues.length - props.limit} lainnya`}</Button>
      )}
    </div>
  );
}

export function UploadPage() {
  const dataset = useStore((s) => s.dataset);
  const status = useStore((s) => s.status.upload);
  const uploadFile = useStore((s) => s.uploadFile);
  const loadSample = useStore((s) => s.loadSample);
  const clear = useStore((s) => s.clearDataset);
  const setPage = useStore((s) => s.setPage);
  const [drag, setDrag] = useState(false);
  const [clientError, setClientError] = useState<string | null>(null);
  const input = useRef<HTMLInputElement | null>(null);

  const handle = async (file: File | null | undefined) => {
    if (!file) return;
    const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
    if (!OK_EXT.includes(ext)) return setClientError("Format tidak didukung. Gunakan file .csv atau .xlsx.");
    if (file.size > MAX_MB * 1024 * 1024) return setClientError(`Ukuran file melebihi ${MAX_MB} MB.`);
    setClientError(null);
    await uploadFile(file);
  };
  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDrag(false);
    void handle(e.dataTransfer.files.item(0));
  };
  const onPick = (e: ChangeEvent<HTMLInputElement>) => {
    void handle(e.target.files?.item(0));
    e.target.value = "";
  };

  const groups = dataset ? groupIssues(dataset.validation.issues) : null;
  const cols: Column<Row>[] = dataset && dataset.preview[0]
    ? Object.keys(dataset.preview[0]).slice(0, 12).map((k) => ({
        key: k, label: k, align: typeof dataset.preview[0]?.[k] === "number" ? "right" : "left",
        render: (r: Row) => { const v = r[k]; return typeof v === "number" ? formatInt(v) : String(v ?? "–"); },
      }))
    : [];

  return (
    <div className="stack">
      <PageHead title="Unggah & Validasi Data" subtitle="Unggah laporan keuangan (CSV atau Excel). Sistem memeriksa skema, tipe data, missing value, duplikat, identitas neraca, dan outlier sebelum analisis." />
      <Card>
        <div
          className={`dropzone ${drag ? "drag" : ""}`} role="button" tabIndex={0} aria-label="Area unggah file"
          onClick={() => input.current?.click()} onKeyDown={(e) => { if (e.key === "Enter") input.current?.click(); }}
          onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)} onDrop={onDrop}
        >
          <h3>Seret file ke sini atau klik untuk memilih</h3>
          <p className="muted small">.csv atau .xlsx · maksimal {MAX_MB} MB · satuan nilai: Rupiah juta · kolom wajib mengikuti skema data</p>
          <input ref={input} type="file" accept=".csv,.xlsx" hidden onChange={onPick} data-testid="file-input" />
        </div>
        <div className="row" style={{ marginTop: 14 }}>
          <span className="muted small">atau coba cepat:</span>
          <Button variant="ghost" size="sm" disabled={status.loading} onClick={() => void loadSample("bersih")}>Data sampel bersih</Button>
          <Button variant="ghost" size="sm" disabled={status.loading} onClick={() => void loadSample("kotor")}>Data sampel &quot;kotor&quot; (ada missing &amp; outlier)</Button>
        </div>
        {status.loading && <Loading text="Membaca dan memvalidasi data…" />}
        {(clientError ?? status.error) && <div style={{ marginTop: 12 }}><ErrorBox message={clientError ?? status.error ?? ""} /></div>}
      </Card>

      {dataset && groups && (
        <>
          <Card title="Hasil validasi" action={<Badge tone={dataset.validation.is_valid ? "ok" : "error"}>{dataset.validation.is_valid ? "Lolos validasi" : "Tidak lolos"}</Badge>}>
            <DatasetSummary dataset={dataset} />
            <IssueList issues={groups.error} tone="error" limit={5} />
            <IssueList issues={groups.warning} tone="warning" limit={5} />
            <IssueList issues={groups.info} tone="info" limit={3} />
          </Card>
          <Card title="Pratinjau data" sub={`10 baris pertama dari ${formatInt(dataset.nRows)} (menampilkan 12 kolom pertama).`}>
            <DataTable rows={dataset.preview} columns={cols} />
          </Card>
          <div className="row">
            <Button onClick={() => setPage("eda")}>Lanjut ke Eksplorasi Data →</Button>
            <Button variant="ghost" onClick={clear}>Hapus dataset</Button>
          </div>
        </>
      )}
    </div>
  );
}
