import { useEffect } from "react";
import { motion } from "framer-motion";
import { useStore } from "../state/store.ts";
import { Button, Card, StatCard } from "../components/ui/ui.tsx";
import { HeroScene } from "../components/three/HeroScene.tsx";
import { ErrorBoundary } from "../components/layout/ErrorBoundary.tsx";
import { formatPercent } from "../lib/format.ts";

const STEPS = [
  { t: "Unggah & validasi", d: "Periksa skema, missing value, outlier, dan identitas neraca sebelum analisis." },
  { t: "Eksplorasi data", d: "Statistik deskriptif, korelasi, perbandingan fraud vs non-fraud, dan uji Benford." },
  { t: "Feature engineering", d: "Rasio keuangan, delapan indeks Beneish, M-Score, dan Altman Z-Score." },
  { t: "Prediksi", d: "Probabilitas terkalibrasi dengan ambang keputusan berbasis biaya." },
  { t: "Penjelasan", d: "Odds ratio dengan interval kepercayaan dan nilai Shapley per perusahaan." },
  { t: "Evaluasi", d: "CI bootstrap, kalibrasi, uji DeLong, dan decision curve analysis." },
] as const;

export function HomePage() {
  const models = useStore((s) => s.models);
  const modeling = useStore((s) => s.modeling);
  const evaluation = useStore((s) => s.evaluation);
  const ensureModeling = useStore((s) => s.ensureModeling);
  const ensureEvaluation = useStore((s) => s.ensureEvaluation);
  const loadSample = useStore((s) => s.loadSample);
  const loading = useStore((s) => s.status.upload.loading);
  const setPage = useStore((s) => s.setPage);

  useEffect(() => {
    void ensureModeling();
    void ensureEvaluation();
  }, [ensureModeling, ensureEvaluation]);

  const ready = models ? models.model.filter((m) => m.tersedia).length : null;
  const bestPr = modeling ? Math.max(...modeling.uji_holdout.filter((r) => !["m_score", "altman_z2"].includes(r.model)).map((r) => r.pr_auc)) : null;
  const fraudRate = evaluation ? evaluation.fraud_uji / evaluation.n_uji : null;

  const start = async () => {
    if (await loadSample("bersih")) setPage("unggah");
  };

  return (
    <div className="stack">
      <section className="hero">
        <ErrorBoundary fallback={null}><HeroScene /></ErrorBoundary>
        <motion.div className="hero-body" initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }}>
          <h1>Deteksi <em>indikasi fraud</em> laporan keuangan dengan statistika yang dapat dijelaskan</h1>
          <p>Dari validasi data, uji Benford, dan rasio Beneish hingga model terkalibrasi dengan interval kepercayaan, sehingga setiap angka dapat dipertanggungjawabkan.</p>
          <div className="row" style={{ marginTop: 22 }}>
            <Button variant="accent" disabled={loading} onClick={() => void start()}>{loading ? "Memuat…" : "Coba dengan data sampel"}</Button>
            <Button variant="light" onClick={() => setPage("evaluasi")}>Lihat evaluasi model</Button>
          </div>
        </motion.div>
      </section>

      <div className="grid cols-4">
        <StatCard label="Model siap" value={ready} hint="dari 4 model prediksi" />
        <StatCard label="PR-AUC terbaik (uji)" value={bestPr === null || !Number.isFinite(bestPr) ? null : bestPr * 100} format={(n) => `${n.toFixed(1).replace(".", ",")}%`} hint="data uji hold-out" />
        <StatCard label="Fraud pada data uji" value={fraudRate === null ? null : fraudRate * 100} format={(n) => `${n.toFixed(1).replace(".", ",")}%`} hint={evaluation ? `${evaluation.fraud_uji} dari ${evaluation.n_uji} baris` : ""} />
        <StatCard label="Ambang default" value={models?.ambang_default == null ? null : models.ambang_default * 100} format={(n) => `${n.toFixed(1).replace(".", ",")}%`} hint="biaya FN : FP = 10 : 1" />
      </div>

      <Card title="Alur kerja" sub="Enam langkah dari data mentah hingga keputusan yang dapat dijelaskan.">
        <div className="grid cols-3 steps">
          {STEPS.map((s) => (
            <div key={s.t} className="card step"><h3 className="card-title">{s.t}</h3><p className="small muted">{s.d}</p></div>
          ))}
        </div>
      </Card>

      <div className="alert warning">
        <strong>Catatan:</strong> data sampel bersifat <strong>sintetis</strong> untuk pengembangan dan demo. Hasil statistik untuk skripsi harus berasal dari data perusahaan sebenarnya
        ({fraudRate === null ? "proporsi fraud sampel ±6%" : `proporsi fraud sampel ${formatPercent(fraudRate, 1)}`}).
      </div>
    </div>
  );
}
