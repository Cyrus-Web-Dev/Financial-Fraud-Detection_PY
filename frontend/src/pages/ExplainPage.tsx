import { useEffect, useMemo, useState } from "react";
import { useStore } from "../state/store.ts";
import { Async, Badge, Card, PageHead, Tabs } from "../components/ui/ui.tsx";
import { DataTable } from "../components/ui/DataTable.tsx";
import { Beeswarm, ForestPlot, HBarChart, Heatmap, ScatterChart } from "../components/charts/charts.tsx";
import { agreementMatrix, beeswarmByFeature, oddsForest, shapBars } from "../lib/chartData.ts";
import { percentileRanks } from "../lib/chartData.ts";
import { formatNumber, formatPValue, formatSmart, modelLabel } from "../lib/format.ts";
import type { ExplanationResults, LocalExplanation } from "../lib/types.ts";

const TABS = [
  { id: "odds", label: "Odds ratio" }, { id: "shap", label: "SHAP global" }, { id: "sebaran", label: "Sebaran SHAP" },
  { id: "dependence", label: "Dependence" }, { id: "importance", label: "Importance & kesepakatan" }, { id: "lokal", label: "Kasus lokal" }, { id: "banding", label: "Interpretabilitas" },
] as const;
type TabId = (typeof TABS)[number]["id"];
const SHAP_MODELS = ["logit_mle", "random_forest", "hist_gb"] as const;

function ModelSelect(props: { value: string; onChange: (v: string) => void; models?: readonly string[] }) {
  return (
    <select className="select" value={props.value} onChange={(e) => props.onChange(e.target.value)} aria-label="Pilih model">
      {(props.models ?? SHAP_MODELS).map((m) => <option key={m} value={m}>{modelLabel(m)}</option>)}
    </select>
  );
}

function OddsTab(props: { ex: ExplanationResults }) {
  const rows = props.ex.interpretasi_odds_ratio;
  return (
    <div className="stack">
      <Card title="Odds ratio per 1 simpangan baku" sub="Garis merah putus-putus: OR = 1 (tanpa efek). Titik gelap = signifikan (α = 5%).">
        <ForestPlot rows={oddsForest(rows)} />
      </Card>
      <Card title="Interpretasi dan kesesuaian dengan teori keuangan" sub="Arah teori adalah hipotesis (mis. Beneish 1999), bukan tanda koefisien estimasi.">
        <DataTable rows={rows} columns={[
          { key: "f", label: "Fitur", render: (r) => <div><strong>{r.label}</strong><div className="mono muted">{r.fitur}</div></div>, sortValue: (r) => r.label },
          { key: "or", label: "OR [CI 95%]", align: "right", render: (r) => `${formatNumber(r.odds_ratio, 2)} [${formatNumber(r.ci_bawah, 2)}; ${formatNumber(r.ci_atas, 2)}]`, sortValue: (r) => r.odds_ratio },
          { key: "p", label: "p-value", align: "right", render: (r) => formatPValue(r.p_value), sortValue: (r) => r.p_value },
          { key: "k", label: "Teori", render: (r) => <Badge tone={r.kesesuaian === "sesuai teori" ? "ok" : r.kesesuaian.startsWith("BERLAWANAN") ? "high" : "neutral"}>{r.kesesuaian}</Badge> },
          { key: "i", label: "Interpretasi", render: (r) => <span className="small">{r.interpretasi}</span> },
        ]} />
      </Card>
    </div>
  );
}

function ShapTab(props: { ex: ExplanationResults }) {
  const [m, setM] = useState<string>("random_forest");
  const labels = useMemo(() => Object.fromEntries(props.ex.interpretasi_odds_ratio.map((r) => [r.fitur, r.label])), [props.ex]);
  const bars = shapBars(props.ex.shap_global[m] ?? [], labels);
  return (
    <Card title="Kepentingan global (rata-rata |SHAP|)" sub={`Porsi kontribusi per fitur. ${props.ex.catatan_sampel}`} action={<ModelSelect value={m} onChange={setM} />}>
      <HBarChart data={bars} format={(n) => `${formatNumber(n, 1)}%`} colorFn={() => "#12305a"} />
      <p className="small muted">Satuan atribusi: {props.ex.shap_satuan[m] ?? "-"}. Bandingkan peringkat dan porsi antar model, bukan besaran mentah.</p>
    </Card>
  );
}

function SwarmTab(props: { ex: ExplanationResults }) {
  const [m, setM] = useState<string>("random_forest");
  const labels = useMemo(() => Object.fromEntries(props.ex.interpretasi_odds_ratio.map((r) => [r.fitur, r.label])), [props.ex]);
  const order = (props.ex.shap_global[m] ?? []).slice().sort((a, b) => b.mean_abs_shap - a.mean_abs_shap).map((r) => r.fitur);
  const groups = beeswarmByFeature(props.ex.beeswarm[m] ?? [], order).map((g) => ({ label: labels[g.fitur] ?? g.fitur, points: g.points.map((p) => ({ shap: p.shap, t: p.persentil_nilai })) }));
  return (
    <Card title="Sebaran nilai SHAP" sub="Setiap titik = satu perusahaan. Posisi horizontal = dampak pada prediksi; warna = nilai fitur (biru rendah, merah tinggi)." action={<ModelSelect value={m} onChange={setM} />}>
      <Beeswarm groups={groups} xLabel={`Nilai SHAP (${props.ex.shap_satuan[m] ?? ""})`} />
    </Card>
  );
}

function DependenceTab(props: { ex: ExplanationResults }) {
  const [m, setM] = useState<string>("random_forest");
  const feats = Object.keys(props.ex.dependence[m] ?? {});
  const [f, setF] = useState<string>(feats[0] ?? "");
  const cur = props.ex.dependence[m]?.[f] ?? props.ex.dependence[m]?.[feats[0] ?? ""];
  const t = cur?.nilai_warna ? percentileRanks(cur.nilai_warna) : undefined;
  const points = cur ? cur.x.map((x, i) => ({ x, y: cur.shap[i] ?? 0, ...(t ? { t: t[i] ?? 0.5 } : {}) })) : [];
  return (
    <Card title="Dependence plot" sub="Hubungan nilai fitur dengan dampaknya. Warna menunjukkan fitur yang paling berinteraksi (bila ada)."
      action={<div className="row"><ModelSelect value={m} onChange={(v) => { setM(v); setF(Object.keys(props.ex.dependence[v] ?? {})[0] ?? ""); }} /><select className="select" value={cur?.fitur ?? ""} onChange={(e) => setF(e.target.value)} aria-label="Pilih fitur">{Object.keys(props.ex.dependence[m] ?? {}).map((k) => <option key={k} value={k}>{k}</option>)}</select></div>}>
      {cur ? <ScatterChart points={points} xLabel={cur.fitur} yLabel="Nilai SHAP" zeroLine /> : <div className="muted">Tidak ada data.</div>}
      {cur?.warna_fitur && <p className="small muted">Warna: persentil nilai <span className="mono">{cur.warna_fitur}</span>.</p>}
    </Card>
  );
}

function ImportanceTab(props: { ex: ExplanationResults }) {
  const [m, setM] = useState<string>("random_forest");
  const perm = (props.ex.permutation_importance[m] ?? []).map((r) => ({ label: r.fitur, value: r.penurunan_mean }));
  const agree = agreementMatrix(props.ex.kesepakatan_peringkat.spearman);
  return (
    <div className="grid cols-2">
      <Card title="Permutation importance" sub="Penurunan PR-AUC data uji saat fitur diacak." action={<ModelSelect value={m} onChange={setM} />}>
        <HBarChart data={perm} format={(n) => formatNumber(n, 3)} colorFn={() => "#2f6fde"} labelWidth={150} />
      </Card>
      <Card title="Kesepakatan peringkat (Spearman)" sub="Antar model dan antar metode (SHAP vs permutation).">
        <Heatmap labels={agree.labels} values={agree.values} />
      </Card>
    </div>
  );
}

function LocalTab(props: { ex: ExplanationResults }) {
  const cases = props.ex.penjelasan_lokal;
  const [i, setI] = useState(0);
  const c = cases[i];
  if (!c) return <div className="muted">Tidak ada kasus lokal.</div>;
  const view = (e: LocalExplanation, title: string) => (
    <Card title={title} sub={`Nilai dasar ${formatNumber(e.nilai_dasar, 3)} → prediksi ${formatNumber(e.prediksi, 3)} (${e.satuan})`}>
      <HBarChart data={e.kontribusi.slice(0, 8).map((k) => ({ label: k.label, value: k.kontribusi }))} format={(n) => formatSmart(n)} colorFn={(b) => (b.value >= 0 ? "#d7263d" : "#1a9a68")} />
      <ul className="small" style={{ paddingLeft: 18 }}>{e.kalimat.map((s) => <li key={s}>{s}</li>)}</ul>
    </Card>
  );
  return (
    <div className="stack">
      <div className="row"><span className="label">Kasus uji:</span>
        <select className="select" value={i} onChange={(e) => setI(Number(e.target.value))} aria-label="Pilih kasus">{cases.map((k, n) => <option key={n} value={n}>{k.company_id} · {k.tahun} ({k.label_sebenarnya ? "fraud" : "non-fraud"})</option>)}</select>
        <Badge tone={c.label_sebenarnya ? "high" : "low"}>label sebenarnya: {c.label_sebenarnya ? "fraud" : "non-fraud"}</Badge>
      </div>
      <div className="grid cols-2">{view(c.random_forest, "Random Forest (probabilitas)")}{view(c.regresi_logistik, "Regresi Logistik (log-odds)")}</div>
    </div>
  );
}

export function ExplainPage() {
  const ex = useStore((s) => s.explanation);
  const status = useStore((s) => s.status.explanation);
  const ensure = useStore((s) => s.ensureExplanation);
  const [tab, setTab] = useState<TabId>("odds");
  useEffect(() => { void ensure(); }, [ensure]);
  return (
    <div className="stack">
      <PageHead title="Penjelasan Model" subtitle="Mengapa model memberi skor tertentu? Odds ratio untuk model parametrik, nilai Shapley eksak untuk model nonlinear, dan kaitannya dengan teori keuangan." />
      <Async state={status} ready={ex !== null} onRetry={() => void ensure()} loadingText="Memuat hasil penjelasan…">
        {ex && (
          <>
            <Tabs tabs={TABS} active={tab} onChange={setTab} />
            {tab === "odds" && <OddsTab ex={ex} />}
            {tab === "shap" && <ShapTab ex={ex} />}
            {tab === "sebaran" && <SwarmTab ex={ex} />}
            {tab === "dependence" && <DependenceTab ex={ex} />}
            {tab === "importance" && <ImportanceTab ex={ex} />}
            {tab === "lokal" && <LocalTab ex={ex} />}
            {tab === "banding" && (
              <Card title="Perbandingan tingkat interpretabilitas">
                <DataTable rows={ex.perbandingan_interpretabilitas} columns={[
                  { key: "m", label: "Model", render: (r) => <strong>{r.model}</strong> },
                  { key: "t", label: "Tingkat", render: (r) => <Badge tone={r.tingkat === "tinggi" ? "ok" : r.tingkat === "sedang" ? "warning" : "high"}>{r.tingkat}</Badge> },
                  { key: "b", label: "Bentuk penjelasan", render: (r) => r.bentuk_penjelasan },
                  { key: "c", label: "Catatan", render: (r) => <span className="small">{r.catatan}</span> },
                ]} />
              </Card>
            )}
          </>
        )}
      </Async>
    </div>
  );
}
