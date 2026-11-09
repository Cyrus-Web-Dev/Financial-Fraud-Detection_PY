import { useEffect, useState } from "react";
import { useStore } from "../state/store.ts";
import { Async, Badge, Card, KeyValue, PageHead, Tabs } from "../components/ui/ui.tsx";
import { DataTable } from "../components/ui/DataTable.tsx";
import { BarChart, HBarChart, LineChart, ReliabilityChart, PALETTE } from "../components/charts/charts.tsx";
import { calibrationPoints, dcaSeries, metricBars, prSeries, rocSeries } from "../lib/chartData.ts";
import { formatInt, formatNumber, formatPercent, formatPValue, modelLabel } from "../lib/format.ts";
import type { BootRow, EvaluationResults, ModelingResults } from "../lib/types.ts";

const TABS = [
  { id: "ringkasan", label: "Ringkasan metrik" }, { id: "kurva", label: "Kurva ROC & PR" }, { id: "kalibrasi", label: "Kalibrasi" },
  { id: "ambang", label: "Ambang & biaya" }, { id: "dca", label: "Decision curve" }, { id: "banding", label: "Perbandingan model" },
  { id: "diagnostik", label: "Diagnostik regresi logistik" }, { id: "pelatihan", label: "Pelatihan & validasi" },
] as const;
type TabId = (typeof TABS)[number]["id"];
const ALL = ["logreg", "logit_mle", "random_forest", "hist_gb", "isolation_forest", "m_score"] as const;
const PROB = ["logreg", "logit_mle", "random_forest", "hist_gb"] as const;
const colorOf = (m: string) => PALETTE[Math.max(0, ALL.indexOf(m as (typeof ALL)[number])) % PALETTE.length] ?? "#12305a";

function ModelSelect(props: { value: string; onChange: (v: string) => void; models?: readonly string[] }) {
  return (
    <select className="select" value={props.value} onChange={(e) => props.onChange(e.target.value)} aria-label="Pilih model">
      {(props.models ?? PROB).map((m) => <option key={m} value={m}>{modelLabel(m)}</option>)}
    </select>
  );
}

function SummaryTab(props: { ev: EvaluationResults }) {
  const { ev } = props;
  const names = ALL.filter((m) => ev.metrik_uji_bootstrap.some((b) => b.model === m));
  const get = (rows: BootRow[], m: string, k: string) => rows.find((r) => r.model === m && r.metrik === k);
  const ci = (b?: BootRow) => (b ? `${formatNumber(b.estimasi, 3)} [${formatNumber(b.ci_bawah, 3)}; ${formatNumber(b.ci_atas, 3)}]` : "–");
  return (
    <div className="stack">
      <div className="alert info">Data uji: <strong>{formatInt(ev.n_uji)}</strong> baris dengan <strong>{formatInt(ev.fraud_uji)}</strong> fraud. CI 95% bootstrap per perusahaan ({formatInt(ev.n_bootstrap)} replikasi).
        Dengan fraud sesedikit ini, selisih antar model yang kecil berada dalam derau statistik.</div>
      <div className="grid cols-3">
        {([["pr_auc", "PR-AUC"], ["roc_auc", "ROC-AUC"], ["recall_at_10", "Recall@10%"]] as const).map(([k, label]) => (
          <Card key={k} title={label} sub="Estimasi dan CI 95%"><BarChart data={metricBars(ev.metrik_uji_bootstrap, k, names).map((b) => ({ ...b, label: modelLabel(b.label).replace(" (inferensial)", "").replace(" (L2)", " L2") }))} height={300} yDomain={[0, 1]} format={(n) => formatNumber(n, 2)} color="#2f6fde" /></Card>
        ))}
      </div>
      <Card title="Tabel metrik uji" sub="Format: estimasi [CI bawah; CI atas]. Beneish M-Score adalah benchmark berbasis teori.">
        <DataTable rows={names} columns={[
          { key: "m", label: "Model", render: (m) => <strong>{modelLabel(m)}</strong> },
          { key: "pr", label: "PR-AUC", render: (m) => ci(get(ev.metrik_uji_bootstrap, m, "pr_auc")) },
          { key: "roc", label: "ROC-AUC", render: (m) => ci(get(ev.metrik_uji_bootstrap, m, "roc_auc")) },
          { key: "rec", label: "Recall@10%", render: (m) => ci(get(ev.metrik_uji_bootstrap, m, "recall_at_10")) },
        ]} />
      </Card>
    </div>
  );
}

function CurvesTab(props: { ev: EvaluationResults }) {
  const models = ["logreg", "random_forest", "hist_gb", "isolation_forest"].filter((m) => props.ev.kurva[m]);
  const withColor = (s: { name: string; points: [number, number][] }[]) => s.map((x) => ({ ...x, label: modelLabel(x.name), color: colorOf(x.name) }));
  return (
    <div className="grid cols-2">
      <Card title="Kurva ROC" sub="Sensitivitas vs 1 − spesifisitas. Garis putus-putus = acak.">
        <LineChart series={withColor(rocSeries(props.ev.kurva, models))} xLabel="Laju false positive" yLabel="Laju true positive (recall)" xDomain={[0, 1]} yDomain={[0, 1]} diagonal />
      </Card>
      <Card title="Kurva Precision-Recall" sub="Lebih informatif daripada ROC pada fraud yang jarang.">
        <LineChart series={withColor(prSeries(props.ev.kurva, models))} xLabel="Recall" yLabel="Precision" xDomain={[0, 1]} yDomain={[0, 1]} />
      </Card>
    </div>
  );
}

function CalibrationTab(props: { ev: EvaluationResults }) {
  const [m, setM] = useState<string>("random_forest");
  const [v, setV] = useState("mentah");
  const rows = props.ev.kalibrasi.filter((r) => r.model === m);
  return (
    <div className="stack">
      <div className="row">
        <ModelSelect value={m} onChange={setM} />
        <select className="select" value={v} onChange={(e) => setV(e.target.value)} aria-label="Versi kalibrasi"><option value="mentah">Mentah</option><option value="platt">Platt scaling</option><option value="isotonik">Regresi isotonik</option></select>
      </div>
      <div className="grid cols-2" style={{ gridTemplateColumns: "minmax(0, 1.1fr) minmax(0, 1fr)" }}>
        <Card title="Diagram reliabilitas" sub="Titik dekat garis putus-putus = probabilitas dapat dipercaya. Garis merah = CI 95% Wilson per bin."><ReliabilityChart points={calibrationPoints(props.ev.kurva_kalibrasi[`${m}:${v}`])} /></Card>
        <Card title="Ukuran kalibrasi" sub="Slope ideal = 1; CITL (calibration-in-the-large) ideal = 0.">
          <DataTable rows={rows} columns={[
            { key: "v", label: "Versi", render: (r) => r.versi },
            { key: "b", label: "Brier", align: "right", render: (r) => formatNumber(r.brier, 4) },
            { key: "e", label: "ECE", align: "right", render: (r) => formatNumber(r.ece, 4) },
            { key: "s", label: "Slope", align: "right", render: (r) => formatNumber(r.slope, 2) },
            { key: "c", label: "CITL", align: "right", render: (r) => formatNumber(r.citl, 2) },
          ]} />
          <p className="small muted" style={{ marginTop: 10 }}>Kalibrator dilatih pada prediksi out-of-fold data latih dan dievaluasi di data uji. Probabilitas setelah pembobotan kelas/resampling cenderung bias ke atas.</p>
        </Card>
      </div>
    </div>
  );
}

function ThresholdTab(props: { ev: EvaluationResults }) {
  const [m, setM] = useState<string>("random_forest");
  const costRule = props.ev.ambang_keputusan.find((r) => r.aturan.startsWith("biaya_"))?.aturan ?? "";
  const rows = props.ev.ambang_keputusan.filter((r) => r.model === m);
  const savings = props.ev.ambang_keputusan.filter((r) => r.aturan === costRule && r.penghematan_biaya_persen !== null).map((r) => ({ label: modelLabel(r.model), value: r.penghematan_biaya_persen ?? 0 }));
  return (
    <div className="stack">
      <div className="alert info">Biaya: false negative (fraud lolos) = <strong>{props.ev.biaya.c_fn}</strong>, false positive (audit sia-sia) = <strong>{props.ev.biaya.c_fp}</strong>; ambang Bayes teoretis = <strong>{formatPercent(props.ev.biaya.ambang_bayes, 1)}</strong>. Ambang dipilih di data latih (out-of-fold), lalu diterapkan ke data uji.</div>
      <Card title="Kinerja pada ambang terpilih" action={<ModelSelect value={m} onChange={setM} models={["logreg", "random_forest", "hist_gb", "isolation_forest"]} />}>
        <DataTable rows={rows} columns={[
          { key: "a", label: "Aturan", render: (r) => <Badge tone={r.aturan.startsWith("biaya") ? "info" : "neutral"}>{r.aturan}</Badge> },
          { key: "t", label: "Ambang", align: "right", render: (r) => formatNumber(r.threshold, 3) },
          { key: "r", label: "Recall", align: "right", render: (r) => formatPercent(r.recall, 1) },
          { key: "p", label: "Precision", align: "right", render: (r) => formatPercent(r.precision, 1) },
          { key: "s", label: "Spesifisitas", align: "right", render: (r) => formatPercent(r.specificity, 1) },
          { key: "f", label: "F1", align: "right", render: (r) => formatNumber(r.f1, 3) },
          { key: "d", label: "% ditandai", align: "right", render: (r) => formatPercent(r.persen_ditandai, 1) },
          { key: "b", label: "Biaya/obs", align: "right", render: (r) => `${formatNumber(r.biaya_per_observasi, 3)} (tanpa model ${formatNumber(r.biaya_tanpa_model_per_observasi, 3)})` },
          { key: "h", label: "Hemat", align: "right", render: (r) => (r.penghematan_biaya_persen === null ? "–" : `${formatNumber(r.penghematan_biaya_persen, 0)}%`) },
        ]} />
      </Card>
      <Card title="Penghematan biaya dibanding tanpa model" sub={`Aturan ${costRule}, data uji.`}><HBarChart data={savings} format={(n) => `${formatNumber(n, 0)}%`} colorFn={() => "#1a9a68"} labelWidth={190} /></Card>
    </div>
  );
}

function DcaTab(props: { ev: EvaluationResults }) {
  const [m, setM] = useState<string>("random_forest");
  const series = dcaSeries(props.ev.decision_curve[m]).map((s, i) => ({ ...s, color: ["#12305a", "#d7263d", "#9aa8c0"][i] ?? "#12305a", dashed: i > 0 }));
  return (
    <Card title="Decision curve analysis" sub="Net benefit: model berguna di ambang tempat kurvanya melampaui 'tandai semua' dan 'tidak tandai siapa pun' (0)."
      action={<ModelSelect value={m} onChange={setM} />}>
      <LineChart series={series} xLabel="Ambang probabilitas" yLabel="Net benefit" xFormat={(n) => formatPercent(n, 0)} />
      <p className="small muted">Proporsi ambang 2–30% tempat model unggul: {formatPercent(props.ev.dca_proporsi_ambang_berguna[m] ?? null, 0)}.</p>
    </Card>
  );
}

function CompareTab(props: { ev: EvaluationResults }) {
  const pairLabel = (a: string, b: string) => `${modelLabel(a)} vs ${modelLabel(b)}`;
  return (
    <div className="stack">
      <Card title="Uji berpasangan AUC dan PR-AUC" sub="DeLong (AUC) dan bootstrap berpasangan per perusahaan (PR-AUC), dengan koreksi Holm untuk 6 perbandingan. DeLong mengasumsikan independensi sehingga cenderung terlalu berani pada data panel.">
        <DataTable rows={props.ev.perbandingan_model} columns={[
          { key: "p", label: "Pasangan", render: (r) => pairLabel(r.model_a, r.model_b) },
          { key: "d", label: "Δ AUC", align: "right", render: (r) => formatNumber(r.selisih_auc, 3) },
          { key: "pd", label: "p DeLong (Holm)", align: "right", render: (r) => formatPValue(r.p_delong_holm), sortValue: (r) => r.p_delong_holm },
          { key: "dp", label: "Δ PR-AUC [CI]", align: "right", render: (r) => `${formatNumber(r.selisih_pr_auc, 3)} [${formatNumber(r.pr_ci_bawah, 3)}; ${formatNumber(r.pr_ci_atas, 3)}]` },
          { key: "pb", label: "p bootstrap (Holm)", align: "right", render: (r) => formatPValue(r.p_bootstrap_pr_holm), sortValue: (r) => r.p_bootstrap_pr_holm },
          { key: "s", label: "", render: (r) => (r.p_delong_holm < 0.05 ? <Badge tone="high">berbeda</Badge> : <Badge>tidak berbeda</Badge>) },
        ]} />
      </Card>
      <Card title="McNemar: sensitivitas pada beban kerja sama" sub="Hanya kasus fraud; setiap model menandai 10% teratas. b = hanya A menangkap, c = hanya B menangkap.">
        <DataTable rows={props.ev.mcnemar_sensitivitas_top10} columns={[
          { key: "p", label: "Pasangan", render: (r) => pairLabel(r.model_a, r.model_b) },
          { key: "b", label: "b", align: "right", render: (r) => String(r.b) }, { key: "c", label: "c", align: "right", render: (r) => String(r.c) },
          { key: "pv", label: "p-value", align: "right", render: (r) => formatPValue(r.p_value) }, { key: "m", label: "Metode", render: (r) => <span className="small">{r.metode}</span> },
        ]} />
      </Card>
    </div>
  );
}

function DiagnosticTab(props: { ev: EvaluationResults }) {
  const d = props.ev.diagnostik_logit;
  const vif = Array.isArray(d.vif) ? Object.assign({}, ...d.vif) as Record<string, number> : d.vif;
  const vifBars = Object.entries(vif).map(([k, v]) => ({ label: k, value: v })).sort((a, b) => b.value - a.value);
  const s = d.pengaruh_ringkasan;
  return (
    <div className="stack">
      <div className="grid cols-3">
        <Card title="Hosmer-Lemeshow (latih)"><div className="stat-value num">p = {formatPValue(d.hosmer_lemeshow_latih.p_value)}</div><div className="stat-hint">χ² = {formatNumber(d.hosmer_lemeshow_latih.statistik, 2)}, db = {d.hosmer_lemeshow_latih.dof}</div><Badge tone={d.hosmer_lemeshow_latih.p_value >= 0.05 ? "ok" : "high"}>{d.hosmer_lemeshow_latih.p_value >= 0.05 ? "model cocok" : "tidak cocok"}</Badge></Card>
        <Card title="Hosmer-Lemeshow (uji)"><div className="stat-value num">p = {formatPValue(d.hosmer_lemeshow_uji.p_value)}</div><div className="stat-hint">χ² = {formatNumber(d.hosmer_lemeshow_uji.statistik, 2)}, db = {d.hosmer_lemeshow_uji.dof}</div></Card>
        <Card title="Pengaruh observasi"><KeyValue items={[["Cook's D maks", formatNumber(s["cooks_d_maks"], 3)], ["Cook's D > 4/n", formatInt(s["cooks_d_gt_4n"])], ["|dev| > 2", `${formatNumber(s["persen_resid_deviance_gt_2"], 1)}%`]]} /></Card>
      </div>
      <Card title="Uji Box-Tidwell (linearitas logit)" sub="p kecil (di bawah ambang Bonferroni) berarti hubungan logit-fitur tidak linear.">
        <DataTable rows={d.box_tidwell} columns={[
          { key: "f", label: "Fitur", render: (r) => <span className="mono">{r.fitur}</span> }, { key: "z", label: "z", align: "right", render: (r) => formatNumber(r.z, 2) },
          { key: "p", label: "p-value", align: "right", render: (r) => formatPValue(r.p_value) },
          { key: "l", label: "Linear?", render: (r) => <Badge tone={r.linear ? "ok" : "high"}>{r.linear ? "ya" : "tidak"}</Badge> },
        ]} />
      </Card>
      <div className="grid cols-2">
        <Card title="Multikolinearitas (VIF)" sub="VIF > 10 bermasalah; > 5 patut diwaspadai."><HBarChart data={vifBars} format={(n) => formatNumber(n, 2)} colorFn={(b) => (b.value > 5 ? "#d7263d" : "#2f6fde")} labelWidth={150} /></Card>
        <Card title="Catatan"><p className="small">{d.catatan_residual}</p></Card>
      </div>
    </div>
  );
}

function TrainingTab(props: { mo: ModelingResults }) {
  const { mo } = props;
  return (
    <div className="stack">
      <Card title="Kinerja hold-out (perusahaan tidak pernah terlihat)">
        <DataTable rows={mo.uji_holdout} columns={[
          { key: "l", label: "Model", render: (r) => <strong>{r.label}</strong> }, { key: "s", label: "Strategi", render: (r) => r.strategi },
          { key: "p", label: "PR-AUC", align: "right", render: (r) => formatNumber(r.pr_auc, 3), sortValue: (r) => r.pr_auc },
          { key: "r", label: "ROC-AUC", align: "right", render: (r) => formatNumber(r.roc_auc, 3), sortValue: (r) => r.roc_auc },
          { key: "k", label: "Recall@10%", align: "right", render: (r) => formatNumber(r.recall_at_10, 3) },
        ]} />
      </Card>
      <div className="grid cols-2">
        <Card title="Penanganan imbalance: sebelum vs sesudah" sub="Jumlah kelas pada data latih.">
          <DataTable rows={mo.distribusi_kelas_resampling} columns={[
            { key: "s", label: "Strategi", render: (r) => r.strategi },
            { key: "b", label: "Non-fraud → fraud (sebelum)", render: (r) => `${formatInt(r.non_fraud_sebelum)} → ${formatInt(r.fraud_sebelum)}` },
            { key: "a", label: "Sesudah", render: (r) => `${formatInt(r.non_fraud_sesudah)} → ${formatInt(r.fraud_sesudah)}` },
          ]} />
        </Card>
        <Card title="Strategi imbalance (CV per perusahaan)" sub="PR-AUC rata-rata ± SD pada data latih.">
          <DataTable rows={mo.perbandingan_imbalance} pageSize={10} columns={[
            { key: "m", label: "Model", render: (r) => modelLabel(r.model) }, { key: "s", label: "Strategi", render: (r) => r.strategi },
            { key: "p", label: "PR-AUC", align: "right", render: (r) => `${formatNumber(r.pr_auc_mean, 3)} ± ${formatNumber(r.pr_auc_sd, 3)}`, sortValue: (r) => r.pr_auc_mean },
          ]} />
        </Card>
      </div>
      <div className="grid cols-2">
        <Card title="Nested CV" sub="Estimasi tidak bias bila hiperparameter ikut dipilih dari data."><DataTable rows={Object.entries(mo.nested_cv)} columns={[
          { key: "m", label: "Model", render: ([k]) => modelLabel(k) }, { key: "s", label: "Strategi", render: ([, v]) => v.strategi },
          { key: "p", label: "PR-AUC", align: "right", render: ([, v]) => `${formatNumber(v.pr_auc_mean, 3)} ± ${formatNumber(v.pr_auc_sd, 3)}` }]} /></Card>
        <Card title="Ablasi seleksi fitur" sub="Fitur terpilih vs semua fitur (CV).">{mo.ablasi_fitur ? <DataTable rows={mo.ablasi_fitur} columns={[
          { key: "m", label: "Model", render: (r) => modelLabel(r.model) }, { key: "h", label: "Fitur", render: (r) => `${r.himpunan_fitur} (${r.n_fitur})` },
          { key: "p", label: "PR-AUC", align: "right", render: (r) => formatNumber(r.pr_auc_mean, 3) }]} /> : <div className="muted small">Belum dijalankan (scripts.run_modeling --ablation-only).</div>}</Card>
      </div>
      <Card title="Uji temporal" sub="Latih pada tahun awal, uji pada tahun terakhir. Perusahaan yang sama muncul di kedua sisi, sehingga angka ini lebih optimistis daripada hold-out per perusahaan.">
        <DataTable rows={mo.uji_temporal} columns={[
          { key: "m", label: "Model", render: (r) => modelLabel(r.model) }, { key: "p", label: "PR-AUC", align: "right", render: (r) => formatNumber(r.pr_auc, 3) }, { key: "r", label: "ROC-AUC", align: "right", render: (r) => formatNumber(r.roc_auc, 3) },
        ]} />
      </Card>
      <span className="small muted">Versi library: {Object.entries(mo.versi_library).map(([k, v]) => `${k} ${v}`).join(" · ")}</span>
    </div>
  );
}

export function EvaluationPage() {
  const ev = useStore((s) => s.evaluation);
  const mo = useStore((s) => s.modeling);
  const evStatus = useStore((s) => s.status.evaluation);
  const moStatus = useStore((s) => s.status.modeling);
  const ensureEv = useStore((s) => s.ensureEvaluation);
  const ensureMo = useStore((s) => s.ensureModeling);
  const [tab, setTab] = useState<TabId>("ringkasan");
  useEffect(() => { void ensureEv(); void ensureMo(); }, [ensureEv, ensureMo]);
  return (
    <div className="stack">
      <PageHead title="Evaluasi Model" subtitle="Evaluasi pada data uji hold-out dengan interval kepercayaan, kalibrasi, ambang berbasis biaya, dan uji perbandingan antar model." />
      <Tabs tabs={TABS} active={tab} onChange={setTab} />
      {tab === "pelatihan" ? (
        <Async state={moStatus} ready={mo !== null} onRetry={() => void ensureMo()}>{mo && <TrainingTab mo={mo} />}</Async>
      ) : (
        <Async state={evStatus} ready={ev !== null} onRetry={() => void ensureEv()} loadingText="Memuat hasil evaluasi…">
          {ev && (
            <>
              {tab === "ringkasan" && <SummaryTab ev={ev} />}
              {tab === "kurva" && <CurvesTab ev={ev} />}
              {tab === "kalibrasi" && <CalibrationTab ev={ev} />}
              {tab === "ambang" && <ThresholdTab ev={ev} />}
              {tab === "dca" && <DcaTab ev={ev} />}
              {tab === "banding" && <CompareTab ev={ev} />}
              {tab === "diagnostik" && <DiagnosticTab ev={ev} />}
            </>
          )}
        </Async>
      )}
    </div>
  );
}
