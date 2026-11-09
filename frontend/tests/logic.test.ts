import { test } from "node:test";
import assert from "node:assert/strict";
import {
  companyLabel, featureBase, filterPredictions, groupIssues, isOutsideTrainingRange, paginate, sliderBounds, sliderStep, sortRows,
  summarizePredictions, thresholdMarks, uniqueSorted,
} from "../src/state/logic.ts";
import { fixture } from "./helpers.ts";
import type { FeatureTable, ModelDetail, PredictionResult, PredictionRow, UploadResult } from "../src/lib/types.ts";

const pred = fixture<PredictionResult>("predict_dataset");

test("ringkasan prediksi konsisten dengan respons API", () => {
  const s = summarizePredictions(pred.hasil);
  assert.equal(s.total, pred.n_baris); assert.equal(s.flagged, pred.n_ditandai); assert.equal(s.tinggi + s.sedang + s.rendah, s.total);
  assert.equal(s.tinggi, s.flagged);                                                            // Tinggi <=> p >= ambang
  assert.ok(s.maxProb >= s.meanProb && s.meanProb > 0);
  assert.deepEqual(summarizePredictions([]), { total: 0, tinggi: 0, sedang: 0, rendah: 0, flagged: 0, meanProb: 0, maxProb: 0 });
});

test("filter prediksi", () => {
  const all = { query: "", onlyFlagged: false, risk: "semua" as const };
  assert.equal(filterPredictions(pred.hasil, all).length, pred.n_baris);
  assert.equal(filterPredictions(pred.hasil, { ...all, onlyFlagged: true }).length, pred.n_ditandai);
  const id = pred.hasil[5]!.company_id;
  assert.ok(filterPredictions(pred.hasil, { ...all, query: ` ${id.toLowerCase()} ` }).every((r) => r.company_id === id));
  assert.ok(filterPredictions(pred.hasil, { ...all, risk: "Sedang" }).every((r) => r.tingkat_risiko === "Sedang"));
  assert.equal(filterPredictions(pred.hasil, { ...all, query: "zzz-tidak-ada" }).length, 0);
  assert.ok(filterPredictions(pred.hasil, { ...all, query: "manufaktur" }).every((r) => r.sektor === "Manufaktur"));
});

test("sortRows stabil, null di akhir, angka vs teks", () => {
  const rows = [{ k: 3, n: "b" }, { k: null, n: "x" }, { k: 1, n: "a" }, { k: 3, n: "c" }, { k: NaN, n: "y" }];
  assert.deepEqual(sortRows(rows, (r) => r.k, "asc").map((r) => r.n), ["a", "b", "c", "x", "y"]);
  assert.deepEqual(sortRows(rows, (r) => r.k, "desc").map((r) => r.n), ["b", "c", "a", "x", "y"]);   // null tetap di akhir; seri stabil
  assert.deepEqual(sortRows(rows, (r) => r.n, "asc").map((r) => r.n), ["a", "b", "c", "x", "y"]);
  const before = rows.map((r) => r.n); sortRows(rows, (r) => r.k); assert.deepEqual(rows.map((r) => r.n), before);   // tidak memutasi
  const top = sortRows(pred.hasil, (r) => r.probabilitas, "desc");
  assert.ok(top.every((r, i) => i === 0 || top[i - 1]!.probabilitas >= r.probabilitas));
});

test("paginate membatasi halaman dengan aman", () => {
  const xs = Array.from({ length: 23 }, (_, i) => i);
  assert.deepEqual(paginate(xs, 1, 10).items, xs.slice(0, 10)); assert.equal(paginate(xs, 3, 10).items.length, 3);
  assert.equal(paginate(xs, 99, 10).page, 3); assert.equal(paginate(xs, -5, 10).page, 1); assert.equal(paginate(xs, 2.7, 10).page, 2);
  assert.deepEqual(paginate([], 1, 10), { items: [], page: 1, pages: 1, total: 0 });
  assert.equal(paginate(xs, 1, 0).items.length, 1);
});

test("pengelompokan isu validasi dari respons unggah asli", () => {
  const up = fixture<UploadResult>("upload_sample");
  const g = groupIssues(up.validasi.issues);
  assert.equal(g.error.length, up.validasi.n_errors); assert.equal(g.warning.length, up.validasi.n_warnings);
  assert.equal(g.error.length + g.warning.length + g.info.length, up.validasi.issues.length);
  assert.deepEqual(uniqueSorted([3, 1, 2, 3, 1]), [1, 2, 3]); assert.deepEqual(uniqueSorted(["b", "a", "b"]), ["a", "b"]);
});

test("featureBase dan label perusahaan memakai tabel fitur asli", () => {
  const ft = fixture<FeatureTable>("features");
  const detail = fixture<ModelDetail>("model_detail_logit");
  const sel = detail.rentang_fitur.map((r) => r.fitur);
  const base = featureBase(ft.baris[0]!, sel);
  assert.ok(base !== null && Object.keys(base).length === sel.length && Object.values(base).every(Number.isFinite));
  assert.equal(featureBase({ ...ft.baris[0]!, dsri: null }, sel), null);
  assert.equal(featureBase({ ...ft.baris[0]!, dsri: "x" }, sel), null);
  assert.equal(companyLabel({ company_id: "PRS-0001", tahun: 2023, sektor: "Jasa" }), "PRS-0001 · 2023 · Jasa");
  assert.equal(companyLabel({}), "? · ? · -");
});

test("slider what-if: langkah, batas, ekstrapolasi", () => {
  const ranges = fixture<ModelDetail>("model_detail_logit").rentang_fitur;
  for (const r of ranges) {
    const step = sliderStep(r); const [lo, hi] = sliderBounds(r);
    assert.ok(step > 0 && Number.isFinite(step), r.fitur); assert.ok(lo < r.median && r.median < hi, r.fitur);
    assert.ok(lo >= r.min - (r.p99 - r.p1) && hi <= r.max + (r.p99 - r.p1), r.fitur);
    assert.ok((hi - lo) / step >= 20 && (hi - lo) / step <= 2000, `${r.fitur} jumlah langkah`);
  }
  assert.equal(sliderStep({ p1: 1, p99: 1 }), 0.01);
  assert.equal(isOutsideTrainingRange(5, { min: 0, max: 4 }), true); assert.equal(isOutsideTrainingRange(2, { min: 0, max: 4 }), false);
});

test("tanda ambang terurut dan menyertakan default", () => {
  const m = thresholdMarks(0.0909);
  assert.deepEqual(m.map((x) => x.value), [0.05, 0.0909, 0.2, 0.5]); assert.equal(m[1]!.label, "default");
  assert.equal(thresholdMarks(null).length, 3); assert.equal(thresholdMarks(NaN).length, 3);
});

test("tipe PredictionRow dari fixture memiliki semua bidang yang dipakai UI", () => {
  const r: PredictionRow = pred.hasil[0]!;
  for (const k of ["company_id", "tahun", "sektor", "skor_model", "probabilitas", "ditandai", "tingkat_risiko", "m_score", "m_flag", "altman_z2_zona"]) assert.ok(k in r, k);
  assert.ok(r.probabilitas > 0 && r.probabilitas < 1);
});

import { probabilityHistogram, reclassify, riskLevel } from "../src/state/logic.ts";

test("reclassify pada ambang asli sama persis dengan hasil backend", () => {
  const again = reclassify(pred.hasil, pred.ambang);
  assert.deepEqual(again.map((r) => [r.ditandai, r.tingkat_risiko]), pred.hasil.map((r) => [r.ditandai, r.tingkat_risiko]));
  assert.deepEqual(again.map((r) => r.probabilitas), pred.hasil.map((r) => r.probabilitas));       // probabilitas tidak berubah
});

test("reclassify: ambang lebih rendah menandai lebih banyak; tidak memutasi input", () => {
  const before = JSON.stringify(pred.hasil[0]);
  const lo = reclassify(pred.hasil, 0.02).filter((r) => r.ditandai).length;
  const mid = reclassify(pred.hasil, 0.09).filter((r) => r.ditandai).length;
  const hi = reclassify(pred.hasil, 0.5).filter((r) => r.ditandai).length;
  assert.ok(lo > mid && mid > hi && hi >= 0);
  assert.equal(JSON.stringify(pred.hasil[0]), before);
  assert.deepEqual(["Tinggi", "Tinggi", "Sedang", "Sedang", "Rendah"].map((x) => x), [0.2, 0.1, 0.06, 0.05, 0.049].map((p) => riskLevel(p, 0.1)));
});

test("histogram probabilitas mencakup semua baris", () => {
  const h = probabilityHistogram(pred.hasil);
  assert.equal(h.length, 6); assert.equal(h.reduce((s, b) => s + b.value, 0), pred.hasil.length);
  assert.equal(probabilityHistogram([{ ...pred.hasil[0]!, probabilitas: 1 }])[5]!.value, 1);       // p = 1 masuk bin terakhir
  assert.equal(probabilityHistogram([{ ...pred.hasil[0]!, probabilitas: 0 }])[0]!.value, 1);
});
