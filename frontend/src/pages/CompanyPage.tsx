import { useEffect, useMemo, useRef, useState } from "react";
import { useStore } from "../state/store.ts";
import { Async, Badge, Button, Card, ErrorBox, Loading, NeedDataset, PageHead } from "../components/ui/ui.tsx";
import { FactorList } from "../components/ui/FactorList.tsx";
import { RiskGauge3D } from "../components/three/RiskGauge3D.tsx";
import { ErrorBoundary } from "../components/layout/ErrorBoundary.tsx";
import { errorMessage } from "../lib/api.ts";
import { companyLabel, featureBase, isOutsideTrainingRange, sliderBounds, sliderStep } from "../state/logic.ts";
import { formatDeltaPoints, formatProb, formatSmart, riskTone } from "../lib/format.ts";
import type { FeatureRange, PredictionResult, WhatIfResult } from "../lib/types.ts";

export function CompanyPage() {
  const api = useStore((s) => s.api);
  const models = useStore((s) => s.models);
  const model = useStore((s) => s.model);
  const setModel = useStore((s) => s.setModel);
  const threshold = useStore((s) => s.threshold);
  const data = useStore((s) => s.companyData);
  const status = useStore((s) => s.status.company);
  const fetchCompany = useStore((s) => s.fetchCompanyData);
  const loadDetail = useStore((s) => s.loadModelDetail);
  const detail = useStore((s) => s.modelDetails[s.model]);
  const dataset = useStore((s) => s.dataset);

  const [idx, setIdx] = useState(0);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [overrides, setOverrides] = useState<Record<string, number>>({});
  const [wi, setWi] = useState<WhatIfResult | null>(null);
  const [wiBusy, setWiBusy] = useState(false);
  const [wiErr, setWiErr] = useState<string | null>(null);
  const seq = useRef(0);

  useEffect(() => { void fetchCompany(); }, [fetchCompany, dataset?.id]);
  useEffect(() => { void loadDetail(model); }, [loadDetail, model]);
  useEffect(() => { setResult(null); setOverrides({}); setWi(null); }, [idx, model, dataset?.id]);

  const ranges: FeatureRange[] = detail?.rentang_fitur ?? [];
  const rawRow = data?.preview.baris[idx];
  const base = useMemo(() => {
    const row = data?.features.baris[idx];
    return row && ranges.length ? featureBase(row, ranges.map((r) => r.fitur)) : null;
  }, [data, idx, ranges]);

  const analyse = async () => {
    if (!rawRow) return;
    setBusy(true); setErr(null);
    try {
      setResult(await api.predictRows({ rows: [rawRow], model, threshold, explain: true }));
    } catch (e) {
      setErr(errorMessage(e));
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    if (!base || Object.keys(overrides).length === 0) { setWi(null); setWiErr(null); return; }
    const id = ++seq.current;
    const timer = setTimeout(() => {
      setWiBusy(true);
      api.whatIf({ model, features: base, overrides, threshold })
        .then((r) => { if (id === seq.current) { setWi(r); setWiErr(null); } })
        .catch((e: unknown) => { if (id === seq.current) setWiErr(errorMessage(e)); })
        .finally(() => { if (id === seq.current) setWiBusy(false); });
    }, 350);
    return () => clearTimeout(timer);
  }, [api, base, overrides, model, threshold]);

  const row = result?.hasil[0];
  return (
    <div className="stack">
      <PageHead title="Analisis Perusahaan" subtitle="Pilih satu laporan keuangan, lihat faktor yang mendorong skor risikonya, lalu ubah rasio untuk melihat dampaknya (what-if)." />
      <NeedDataset>
        <Async state={status} ready={data !== null} onRetry={() => void fetchCompany()} loadingText="Menyiapkan data perusahaan…">
          {data && (
            <div className="grid cols-2" style={{ gridTemplateColumns: "minmax(0, 1fr) minmax(0, 1.2fr)" }}>
              <div className="stack">
                <Card title="1. Pilih perusahaan">
                  <div className="field">
                    <label htmlFor="co">Laporan keuangan ({data.preview.baris.length} baris pertama)</label>
                    <select id="co" className="select" value={idx} onChange={(e) => setIdx(Number(e.target.value))}>
                      {data.preview.baris.map((r, i) => <option key={i} value={i}>{companyLabel(r)}</option>)}
                    </select>
                  </div>
                  <div className="field" style={{ marginTop: 10 }}>
                    <label htmlFor="mdl">Model</label>
                    <select id="mdl" className="select" value={model} onChange={(e) => setModel(e.target.value as typeof model)}>
                      {(models?.model ?? []).filter((m) => m.tersedia).map((m) => <option key={m.nama} value={m.nama}>{m.label}</option>)}
                    </select>
                  </div>
                  <div className="row" style={{ marginTop: 14 }}><Button variant="accent" disabled={busy} onClick={() => void analyse()}>{busy ? "Menganalisis…" : "Analisis perusahaan"}</Button></div>
                  {err && <div style={{ marginTop: 10 }}><ErrorBox message={err} /></div>}
                </Card>
                {row && result && (
                  <Card title="Skor risiko" sub={`${row.company_id} · ${row.tahun} · ${row.sektor}`}>
                    <ErrorBoundary fallback={<div className="stat-value">{formatProb(row.probabilitas)}</div>}>
                      <RiskGauge3D probability={row.probabilitas} tone={riskTone(row.tingkat_risiko)} caption={row.tingkat_risiko} />
                    </ErrorBoundary>
                    <div className="row" style={{ justifyContent: "center", gap: 8 }}>
                      <Badge tone={riskTone(row.tingkat_risiko)}>{row.ditandai ? "Ditandai untuk diperiksa" : "Tidak ditandai"}</Badge>
                      <Badge tone={row.m_flag ? "high" : "low"}>Beneish: {row.m_flag ? "manipulator?" : "wajar"}</Badge>
                      {row.altman_z2_zona && <Badge tone="info">Altman Z″: {row.altman_z2_zona}</Badge>}
                    </div>
                    {row.faktor_utama && <><h4 style={{ marginTop: 14 }}>Faktor utama</h4><FactorList factors={row.faktor_utama} unit={result.satuan_penjelasan} /></>}
                  </Card>
                )}
              </div>
              <Card title="2. What-if: ubah rasio keuangan" sub="Geser slider; prediksi dihitung ulang otomatis. Nilai di luar rentang data latih ditandai sebagai ekstrapolasi.">
                {!detail && <Loading text="Memuat rentang fitur…" />}
                {detail && !base && <div className="alert warning">Perusahaan ini memiliki nilai fitur kosong atau tidak terdefinisi, sehingga what-if tidak tersedia.</div>}
                {detail && base && (
                  <>
                    {ranges.map((r) => {
                      const baseV = base[r.fitur] ?? r.median;
                      const cur = overrides[r.fitur] ?? baseV;
                      const [lo0, hi0] = sliderBounds(r);
                      const lo = Math.min(lo0, baseV), hi = Math.max(hi0, baseV);
                      const changed = r.fitur in overrides;
                      return (
                        <div key={r.fitur} className="field" style={{ marginBottom: 12 }}>
                          <label htmlFor={`s-${r.fitur}`}>
                            {r.label} <span className="mono muted">{r.fitur}</span>
                            <span className="num" style={{ float: "right", color: changed ? "var(--red)" : "inherit" }}>{formatSmart(cur)}</span>
                          </label>
                          <input id={`s-${r.fitur}`} className="range" type="range" min={lo} max={hi} step={sliderStep(r)} value={cur}
                            onChange={(e) => setOverrides((o) => ({ ...o, [r.fitur]: Number(e.target.value) }))} />
                          <div className="small muted row between">
                            <span>awal {formatSmart(baseV)} · rentang latih {formatSmart(r.min)} – {formatSmart(r.max)}</span>
                            {changed && <span className="row" style={{ gap: 6 }}>{isOutsideTrainingRange(cur, r) && <Badge tone="warning">ekstrapolasi</Badge>}<Button variant="ghost" size="sm" onClick={() => setOverrides((o) => { const n = { ...o }; delete n[r.fitur]; return n; })}>reset</Button></span>}
                          </div>
                        </div>
                      );
                    })}
                    <div className="row"><Button variant="ghost" size="sm" disabled={Object.keys(overrides).length === 0} onClick={() => setOverrides({})}>Reset semua</Button></div>
                  </>
                )}
                {wiErr && <div style={{ marginTop: 10 }}><ErrorBox message={wiErr} /></div>}
                {wi && (
                  <div style={{ marginTop: 16 }}>
                    <div className="grid cols-3">
                      <div><div className="stat-label">Sebelum</div><div className="stat-value num">{formatProb(wi.sebelum.probabilitas)}</div><Badge tone={riskTone(wi.sebelum.tingkat_risiko)}>{wi.sebelum.tingkat_risiko}</Badge></div>
                      <div><div className="stat-label">Sesudah</div><div className="stat-value num">{formatProb(wi.sesudah.probabilitas)}</div><Badge tone={riskTone(wi.sesudah.tingkat_risiko)}>{wi.sesudah.tingkat_risiko}</Badge></div>
                      <div><div className="stat-label">Perubahan</div><div className={`stat-value num ${wi.selisih_probabilitas > 0 ? "delta-up" : "delta-down"}`} style={{ fontSize: 24 }}>{formatDeltaPoints(wi.selisih_probabilitas, 2)}</div>{wiBusy && <span className="small muted">menghitung…</span>}</div>
                    </div>
                    {wi.peringatan.map((w) => <div key={w} className="alert warning" style={{ marginTop: 8 }}>{w}</div>)}
                  </div>
                )}
              </Card>
            </div>
          )}
        </Async>
      </NeedDataset>
    </div>
  );
}
