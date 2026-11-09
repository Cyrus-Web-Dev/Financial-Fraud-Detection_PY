import { useEffect, useState } from "react";
import { useStore } from "../state/store.ts";
import { Async, Badge, Card, KeyValue, NeedDataset, PageHead, Tabs } from "../components/ui/ui.tsx";
import { DataTable } from "../components/ui/DataTable.tsx";
import type { Column } from "../components/ui/DataTable.tsx";
import { BarChart, HBarChart, Heatmap, LineChart } from "../components/charts/charts.tsx";
import { benfordBars, classBars, heatmapCells } from "../lib/chartData.ts";
import { formatInt, formatNumber, formatPercent, formatPValue, formatSmart } from "../lib/format.ts";
import type { DescRow, EdaSummary, GroupCompare } from "../lib/types.ts";

const TABS = [
  { id: "ringkasan", label: "Distribusi kelas" }, { id: "deskriptif", label: "Statistik deskriptif" }, { id: "korelasi", label: "Korelasi" },
  { id: "benford", label: "Uji Benford" }, { id: "perbandingan", label: "Fraud vs non-fraud" }, { id: "outlier", label: "Outlier" },
] as const;
type TabId = (typeof TABS)[number]["id"];

function ClassTab(props: { eda: EdaSummary }) {
  const { eda } = props;
  if (!eda.distribusi_kelas) return <div className="alert info">Dataset ini tidak memiliki kolom label <span className="mono">is_fraud</span>, sehingga analisis per kelas tidak tersedia.</div>;
  const all = eda.distribusi_kelas[0];
  return (
    <div className="stack">
      {all && (
        <Card title="Ketidakseimbangan kelas" sub="Pada fraud yang jarang, accuracy menyesatkan; gunakan PR-AUC, recall, dan kalibrasi.">
          <div className="grid cols-4">
            <div><div className="stat-label">Observasi</div><div className="stat-value num">{formatInt(all.n)}</div></div>
            <div><div className="stat-label">Fraud</div><div className="stat-value num">{formatInt(all.fraud)}</div></div>
            <div><div className="stat-label">Non-fraud</div><div className="stat-value num">{formatInt(all.non_fraud)}</div></div>
            <div><div className="stat-label">Proporsi fraud</div><div className="stat-value num">{formatPercent(all.proporsi_fraud, 2)}</div><div className="stat-hint">CI 95% Wilson: {formatPercent(all.ci95_bawah, 1)}–{formatPercent(all.ci95_atas, 1)}</div></div>
          </div>
        </Card>
      )}
      <div className="grid cols-2">
        <Card title="Proporsi fraud per tahun" sub="Batang = proporsi; garis = CI 95% Wilson."><BarChart data={classBars(eda.distribusi_per_tahun ?? [])} format={(n) => formatPercent(n, 1)} /></Card>
        <Card title="Proporsi fraud per sektor"><BarChart data={classBars(eda.distribusi_per_sektor ?? [])} format={(n) => formatPercent(n, 1)} color="#12305a" /></Card>
      </div>
    </div>
  );
}

function DescTab(props: { eda: EdaSummary }) {
  const rejected = props.eda.normalitas.filter((r) => r.normal === "tidak").length;
  const cols: Column<DescRow>[] = [
    { key: "kolom", label: "Pos", render: (r) => <span className="mono">{r.kolom}</span>, sortValue: (r) => r.kolom },
    { key: "n", label: "n", align: "right", render: (r) => formatInt(r.n), sortValue: (r) => r.n },
    { key: "miss", label: "Missing", align: "right", render: (r) => `${formatNumber(r.persen_missing, 1)}%`, sortValue: (r) => r.persen_missing },
    { key: "mean", label: "Rata-rata", align: "right", render: (r) => formatSmart(r.mean), sortValue: (r) => r.mean },
    { key: "sd", label: "SD", align: "right", render: (r) => formatSmart(r.sd), sortValue: (r) => r.sd },
    { key: "min", label: "Min", align: "right", render: (r) => formatSmart(r.min), sortValue: (r) => r.min },
    { key: "med", label: "Median", align: "right", render: (r) => formatSmart(r.median), sortValue: (r) => r.median },
    { key: "max", label: "Maks", align: "right", render: (r) => formatSmart(r.max), sortValue: (r) => r.max },
    { key: "skew", label: "Skewness", align: "right", render: (r) => formatNumber(r.skewness, 2), sortValue: (r) => r.skewness },
    { key: "kurt", label: "Kurtosis", align: "right", render: (r) => formatNumber(r.kurtosis, 2), sortValue: (r) => r.kurtosis },
  ];
  return (
    <div className="stack">
      <Card title="Statistik deskriptif" sub="Kurtosis = excess kurtosis (normal = 0). Klik judul kolom untuk mengurutkan.">
        <DataTable rows={props.eda.deskriptif} columns={cols} pageSize={12} />
      </Card>
      <div className="alert info">
        <strong>Normalitas:</strong> {rejected} dari {props.eda.normalitas.length} pos menolak normalitas (Shapiro-Wilk, α = 5%). Distribusi keuangan miring ke kanan, sehingga dipakai uji nonparametrik
        (Mann-Whitney), winsorizing, dan transformasi log.
      </div>
    </div>
  );
}

function CorrTab(props: { eda: EdaSummary }) {
  const h = heatmapCells(props.eda.korelasi);
  const pairs = props.eda.korelasi_tinggi.slice(0, 12);
  return (
    <div className="grid cols-2" style={{ gridTemplateColumns: "minmax(0, 1.4fr) minmax(0, 1fr)" }}>
      <Card title="Matriks korelasi Spearman" sub="Nilai mentah saling berkorelasi kuat karena semuanya mengikuti ukuran perusahaan, sehingga dipakai rasio.">
        <Heatmap labels={h.labels} values={h.values} />
      </Card>
      <Card title="Pasangan berkorelasi tinggi" sub={`|ρ| ≥ 0,8: ${props.eda.korelasi_tinggi.length} pasang (12 teratas).`}>
        <DataTable rows={pairs} columns={[
          { key: "a", label: "Variabel 1", render: (r) => <span className="mono">{r.variabel_1}</span> },
          { key: "b", label: "Variabel 2", render: (r) => <span className="mono">{r.variabel_2}</span> },
          { key: "r", label: "ρ", align: "right", render: (r) => formatNumber(r.korelasi, 3) },
        ]} />
      </Card>
    </div>
  );
}

function BenfordTab(props: { eda: EdaSummary }) {
  const groups = Object.keys(props.eda.benford_per_kelompok);
  const [g, setG] = useState(groups[0] ?? "semua");
  const data = props.eda.benford_per_kelompok[g] ?? props.eda.benford_per_kelompok[groups[0] ?? ""];
  if (!data) return null;
  const first = benfordBars(data.first_digit);
  const last = benfordBars(data.last_digit);
  const tone = data.first_digit.conformity === "Sangat sesuai" || data.first_digit.conformity === "Cukup sesuai" ? "ok" : "warning";
  const pct = (n: number) => formatPercent(n, 0);
  return (
    <div className="stack">
      <div className="row">
        <span className="label">Kelompok:</span>
        <select className="select" value={g} onChange={(e) => setG(e.target.value)} aria-label="Pilih kelompok">
          {groups.map((k) => <option key={k} value={k}>{k.replace("_", "-")} ({formatInt(props.eda.benford_per_kelompok[k]?.n_baris)} baris)</option>)}
        </select>
      </div>
      <div className="grid cols-2">
        <Card title="Digit pertama (Hukum Benford)" sub="Penyimpangan dari Benford bukan bukti fraud, hanya indikasi untuk diperiksa.">
          <LineChart series={[
            { name: "Teramati", label: "Teramati", points: first.map((b): [number, number] => [b.digit, b.observed]), color: "#d7263d" },
            { name: "Benford", label: "Harapan Benford", points: first.map((b): [number, number] => [b.digit, b.expected]), color: "#12305a", dashed: true },
          ]} xLabel="Digit pertama" yLabel="Proporsi" xFormat={(n) => String(n)} xTicks={[1, 2, 3, 4, 5, 6, 7, 8, 9]} yFormat={pct} yDomain={[0, 0.35]} xDomain={[1, 9]} height={280} />
          <KeyValue items={[
            ["Kesesuaian (MAD)", <><Badge tone={tone}>{data.first_digit.conformity ?? "–"}</Badge> MAD = {formatNumber(data.first_digit.mad, 4)}</>],
            ["χ² / p-value", `${formatNumber(data.first_digit.chi2, 1)} (db ${data.first_digit.dof}) / ${formatPValue(data.first_digit.p_value)}`],
            ["Jumlah angka", formatInt(data.first_digit.n)],
          ]} />
          <p className="small muted" style={{ marginTop: 8 }}>Pada n besar, χ² hampir selalu menolak; MAD (ambang Nigrini: 0,006 / 0,012 / 0,015) lebih informatif.</p>
        </Card>
        <Card title="Digit terakhir (keseragaman)" sub="Angka karangan cenderung dibulatkan, sehingga digit 0 dan 5 berlebih.">
          <LineChart series={[
            { name: "Teramati", label: "Teramati", points: last.map((b): [number, number] => [b.digit, b.observed]), color: "#d7263d" },
            { name: "Seragam", label: "Seragam 10%", points: last.map((b): [number, number] => [b.digit, b.expected]), color: "#12305a", dashed: true },
          ]} xLabel="Digit terakhir" yLabel="Proporsi" xFormat={(n) => String(n)} xTicks={[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]} yFormat={pct} yDomain={[0, 0.5]} xDomain={[0, 9]} height={280} />
          <KeyValue items={[
            ["Hasil uji", <Badge tone={(data.last_digit.p_value ?? 1) >= 0.05 ? "ok" : "high"}>{data.last_digit.conformity ?? "–"}</Badge>],
            ["χ² / p-value", `${formatNumber(data.last_digit.chi2, 1)} (db ${data.last_digit.dof}) / ${formatPValue(data.last_digit.p_value)}`],
            ["Jumlah angka", formatInt(data.last_digit.n)],
          ]} />
        </Card>
      </div>
      <Card title="Benford per pos laporan">
        <DataTable rows={props.eda.benford_per_kolom} pageSize={10} columns={[
          { key: "k", label: "Pos", render: (r) => <span className="mono">{r.kolom}</span>, sortValue: (r) => r.kolom },
          { key: "n", label: "n", align: "right", render: (r) => formatInt(r.n) },
          { key: "mad", label: "MAD", align: "right", render: (r) => formatNumber(r.mad, 4), sortValue: (r) => r.mad },
          { key: "p", label: "p-value", align: "right", render: (r) => formatPValue(r.p_value), sortValue: (r) => r.p_value },
          { key: "c", label: "Kesesuaian", render: (r) => r.kesesuaian ?? "–" },
        ]} />
      </Card>
    </div>
  );
}

function CompareTab(props: { eda: EdaSummary }) {
  const rows = props.eda.perbandingan_kelompok;
  if (!rows) return <div className="alert info">Perbandingan fraud vs non-fraud membutuhkan kolom label.</div>;
  const cols: Column<GroupCompare>[] = [
    { key: "v", label: "Variabel", render: (r) => <span className="mono">{r.variabel}</span>, sortValue: (r) => r.variabel },
    { key: "mf", label: "Median fraud", align: "right", render: (r) => formatSmart(r.median_fraud) },
    { key: "mn", label: "Median non-fraud", align: "right", render: (r) => formatSmart(r.median_non_fraud) },
    { key: "e", label: "Efek (r)", align: "right", render: (r) => formatNumber(r.efek_rank_biserial, 3), sortValue: (r) => r.efek_rank_biserial },
    { key: "p", label: "p-value", align: "right", render: (r) => formatPValue(r.p_value), sortValue: (r) => r.p_value },
    { key: "q", label: "q (BH)", align: "right", render: (r) => formatPValue(r.q_value), sortValue: (r) => r.q_value },
    { key: "s", label: "", render: (r) => ((r.q_value ?? 1) < 0.05 ? <Badge tone="high">signifikan</Badge> : <Badge>tidak</Badge>) },
  ];
  const bars = [...rows].filter((r) => r.efek_rank_biserial !== null).sort((a, b) => Math.abs(b.efek_rank_biserial ?? 0) - Math.abs(a.efek_rank_biserial ?? 0)).slice(0, 12)
    .map((r) => ({ label: r.variabel, value: r.efek_rank_biserial ?? 0 }));
  return (
    <div className="grid cols-2" style={{ gridTemplateColumns: "minmax(0, 1.25fr) minmax(0, 1fr)" }}>
      <Card title="Uji Mann-Whitney U" sub="Koreksi Benjamini-Hochberg (FDR). Efek r > 0: nilai fraud cenderung lebih besar."><DataTable rows={rows} columns={cols} pageSize={10} /></Card>
      <Card title="Ukuran efek (rank-biserial)" sub="12 pos dengan efek terbesar."><HBarChart data={bars} format={(n) => formatNumber(n, 2)} colorFn={(b) => (b.value >= 0 ? "#d7263d" : "#2f6fde")} labelWidth={150} /></Card>
    </div>
  );
}

function OutlierTab(props: { eda: EdaSummary }) {
  return (
    <Card title="Outlier (modified z-score berbasis MAD)" sub="Dihitung pada rasio terhadap total aset agar tidak bergantung ukuran perusahaan; ambang |M| > 8.">
      <DataTable rows={props.eda.outlier} empty="Tidak ada outlier ekstrem terdeteksi." columns={[
        { key: "k", label: "Pos", render: (r) => <span className="mono">{r.kolom}</span> },
        { key: "n", label: "Jumlah sel", align: "right", render: (r) => formatInt(r.jumlah_outlier), sortValue: (r) => r.jumlah_outlier },
        { key: "p", label: "% baris", align: "right", render: (r) => `${formatNumber(r.persen, 2)}%` },
      ]} />
    </Card>
  );
}

export function EdaPage() {
  const eda = useStore((s) => s.eda);
  const status = useStore((s) => s.status.eda);
  const fetchEda = useStore((s) => s.fetchEda);
  const dataset = useStore((s) => s.dataset);
  const [tab, setTab] = useState<TabId>("ringkasan");
  useEffect(() => { void fetchEda(); }, [fetchEda, dataset?.id]);
  return (
    <div className="stack">
      <PageHead title="Eksplorasi Data" subtitle="Kenali karakteristik data sebelum pemodelan: ketidakseimbangan kelas, distribusi, korelasi, dan pola digit yang tidak wajar." />
      <NeedDataset>
        <Async state={status} ready={eda !== null} onRetry={() => void fetchEda()} loadingText="Menghitung statistik eksploratif…">
          {eda && (
            <>
              <Tabs tabs={TABS} active={tab} onChange={setTab} />
              {tab === "ringkasan" && <ClassTab eda={eda} />}
              {tab === "deskriptif" && <DescTab eda={eda} />}
              {tab === "korelasi" && <CorrTab eda={eda} />}
              {tab === "benford" && <BenfordTab eda={eda} />}
              {tab === "perbandingan" && <CompareTab eda={eda} />}
              {tab === "outlier" && <OutlierTab eda={eda} />}
            </>
          )}
        </Async>
      </NeedDataset>
    </div>
  );
}
