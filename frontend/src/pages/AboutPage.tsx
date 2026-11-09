import { useEffect } from "react";
import { useStore } from "../state/store.ts";
import { Card, KeyValue, PageHead } from "../components/ui/ui.tsx";

export function AboutPage() {
  const modeling = useStore((s) => s.modeling);
  const evaluation = useStore((s) => s.evaluation);
  const ensureModeling = useStore((s) => s.ensureModeling);
  const ensureEvaluation = useStore((s) => s.ensureEvaluation);
  useEffect(() => { void ensureModeling(); void ensureEvaluation(); }, [ensureModeling, ensureEvaluation]);
  const split = modeling?.split;
  const num = (k: string): string => (split && typeof split[k] === "number" ? String(split[k]) : "–");
  return (
    <div className="stack">
      <PageHead title="Tentang Model" subtitle="Transparansi ilmiah: data, asumsi, keterbatasan, dan potensi bias. Bagian ini penting dibaca sebelum menafsirkan hasil." />
      <Card title="Data dan metodologi">
        <KeyValue items={[
          ["Jenis data", "Panel laporan keuangan (perusahaan × tahun), satuan Rupiah juta"],
          ["Sumber (sampel)", "SINTETIS: dibangkitkan untuk pengembangan dan demo, bukan data perusahaan sebenarnya"],
          ["Pembagian data", split ? `Per perusahaan (stratified): latih ${num("n_latih")} baris, uji ${num("n_uji")} baris, ${num("perusahaan_beririsan")} perusahaan beririsan` : "–"],
          ["Fitur model", modeling ? `${modeling.fitur_terpilih.length} fitur terpilih (VIF ≤ 10, Lasso + stability selection)` : "–"],
          ["Validasi", "CV stratified-group 5-lipatan, nested CV untuk tuning, uji hold-out per perusahaan"],
          ["Biaya keputusan", evaluation ? `FN : FP = ${evaluation.biaya.c_fn} : ${evaluation.biaya.c_fp} (asumsi, dapat diubah)` : "–"],
        ]} />
      </Card>
      <div className="grid cols-2">
        <Card title="Asumsi">
          <div className="prose"><ul>
            <li>Rasio keuangan dari laporan dapat dipercaya sebagai bahan analisis (garbage in, garbage out).</li>
            <li>Regresi logistik mengasumsikan logit linear terhadap fitur (diuji dengan Box-Tidwell) dan tanpa multikolinearitas berat (VIF).</li>
            <li>Probabilitas dikalibrasi dengan Platt scaling pada prediksi out-of-fold.</li>
            <li>Koefisien Beneish (1999) berasal dari data manufaktur AS dan hanya dipakai sebagai pembanding.</li>
          </ul></div>
        </Card>
        <Card title="Keterbatasan dan bias">
          <div className="prose"><ul>
            <li><strong>Data sintetis:</strong> fitur pembulatan digit dikodekan kuat di generator; pada data nyata pengaruhnya akan jauh lebih kecil.</li>
            <li>Jumlah fraud pada data uji kecil, sehingga CI lebar dan perbedaan antar model tidak bermakna.</li>
            <li>Seleksi fitur dilakukan sebelum CV pada data latih, sehingga skor CV sedikit optimistis; gunakan hasil hold-out.</li>
            <li>Label fraud pada data nyata rentan <em>selection bias</em> (hanya fraud yang terdeteksi yang berlabel) dan <em>measurement bias</em>.</li>
            <li>Model tidak menjamin generalisasi ke sektor, periode, atau regulasi yang berbeda.</li>
          </ul></div>
        </Card>
      </div>
      <div className="alert warning"><strong>Peringatan penggunaan:</strong> skor risiko adalah <em>indikasi untuk diperiksa lebih lanjut</em>, bukan bukti fraud. Keputusan audit tetap memerlukan pertimbangan profesional dan bukti tambahan.</div>
      <Card title="Reproducibility">
        <KeyValue items={[
          ["Random seed", "42 (seluruh tahap)"],
          ["Versi library", modeling ? Object.entries(modeling.versi_library).map(([k, v]) => `${k} ${v}`).join(" · ") : "–"],
          ["Perintah", <span className="mono">python -m scripts.run_feature_engineering && python -m scripts.run_modeling && python -m scripts.run_evaluation</span>],
        ]} />
      </Card>
    </div>
  );
}
