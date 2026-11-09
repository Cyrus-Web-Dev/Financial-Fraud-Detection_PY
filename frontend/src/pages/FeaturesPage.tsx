import { useEffect, useMemo } from "react";
import { useStore } from "../state/store.ts";
import { Async, Badge, Card, NeedDataset, PageHead } from "../components/ui/ui.tsx";
import { DataTable } from "../components/ui/DataTable.tsx";
import type { Column } from "../components/ui/DataTable.tsx";
import { formatSmart, isNum } from "../lib/format.ts";
import type { CatalogItem, Row } from "../lib/types.ts";

function SignBadge(props: { sign: number }) {
  return props.sign > 0 ? <Badge tone="high">+ menaikkan risiko</Badge> : props.sign < 0 ? <Badge tone="low">− menurunkan risiko</Badge> : <Badge>arah tidak pasti</Badge>;
}

export function FeaturesPage() {
  const catalog = useStore((s) => s.catalog);
  const table = useStore((s) => s.featureTable);
  const status = useStore((s) => s.status.features);
  const fetchFeatures = useStore((s) => s.fetchFeatures);
  const dataset = useStore((s) => s.dataset);
  useEffect(() => { void fetchFeatures(); }, [fetchFeatures, dataset?.id]);

  const groups = useMemo(() => {
    const m = new Map<string, CatalogItem[]>();
    for (const f of catalog?.fitur ?? []) m.set(f.group, [...(m.get(f.group) ?? []), f]);
    return [...m.entries()];
  }, [catalog]);

  const shown = catalog?.fitur_terpilih.length ? catalog.fitur_terpilih : table?.fitur.slice(0, 8) ?? [];
  const cols: Column<Row>[] = [
    { key: "company_id", label: "Perusahaan", render: (r) => <span className="mono">{String(r["company_id"])}</span> },
    { key: "tahun", label: "Tahun", render: (r) => String(r["tahun"]) },
    ...shown.map((f): Column<Row> => ({ key: f, label: f, align: "right", render: (r) => { const v = r[f]; return isNum(v) ? formatSmart(v) : "–"; }, sortValue: (r) => { const v = r[f]; return isNum(v) ? v : null; } })),
    { key: "m_score", label: "M-Score", align: "right", render: (r) => { const v = r["m_score"]; return isNum(v) ? formatSmart(v) : "–"; } },
    { key: "m_flag", label: "Tanda Beneish", render: (r) => (r["m_flag"] === 1 ? <Badge tone="high">manipulator?</Badge> : <Badge tone="low">wajar</Badge>) },
    { key: "z", label: "Zona Altman Z″", render: (r) => String(r["altman_z2_zone"] ?? "–") },
  ];

  return (
    <div className="stack">
      <PageHead title="Feature Engineering" subtitle="Nilai mentah laporan keuangan diubah menjadi rasio yang tidak bergantung ukuran perusahaan, dilengkapi benchmark berbasis teori." />
      <NeedDataset>
        <Card title="Rasio hasil perhitungan" sub={`20 baris pertama dataset. Kolom fitur adalah fitur yang dipilih model (${shown.length}); M-Score dan Altman Z″ adalah benchmark, bukan fitur model.`}>
          <Async state={status} ready={table !== null} onRetry={() => void fetchFeatures()} loadingText="Menghitung rasio keuangan…">
            {table && <DataTable rows={table.baris} columns={cols} pageSize={10} />}
          </Async>
        </Card>
      </NeedDataset>
      <Card title="Katalog fitur" sub="36 fitur dengan rumus, dasar teori, dan arah hubungan yang diharapkan. Arah adalah hipotesis teori, bukan hasil estimasi.">
        {!catalog ? <div className="muted">Katalog belum tersedia (API belum siap).</div> : (
          <div className="stack">
            {groups.map(([g, items]) => (
              <div key={g}>
                <h4 style={{ margin: "6px 0 8px" }}>{g}</h4>
                <DataTable rows={items} columns={[
                  { key: "l", label: "Fitur", render: (r) => <div><strong>{r.label}</strong> <span className="mono muted">{r.name}</span><div className="small muted">{r.rationale}</div></div> },
                  { key: "f", label: "Rumus", render: (r) => <span className="small">{r.formula}</span> },
                  { key: "s", label: "Arah teori", render: (r) => <SignBadge sign={r.expected_sign} /> },
                  { key: "u", label: "Dipakai model", render: (r) => (r.dipakai_model ? <Badge tone="ok">ya</Badge> : <span className="muted small">tidak</span>) },
                ]} />
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}
