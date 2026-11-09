import { test } from "node:test";
import assert from "node:assert/strict";
import {
  agreementMatrix, benfordBars, calibrationPoints, classBars, dcaSeries, heatmapCells, metricBars, oddsForest, prSeries, rocSeries,
  shapBars, symmetricLimit, beeswarmByFeature,
} from "../src/lib/chartData.ts";
import { processed } from "./helpers.ts";
import type { EdaSummary, EvaluationResults, ExplanationResults } from "../src/lib/types.ts";

const eda = processed<EdaSummary>("eda_summary");
const ev = processed<EvaluationResults>("evaluation_results");
const ex = processed<ExplanationResults>("explanation_results");
const need = (x: unknown, t: { skip: (m?: string) => void }) => { if (!x) t.skip("artefak backend belum dibuat"); return !!x; };

test("EDA: Benford, heatmap, distribusi kelas cocok dengan data asli", (t) => {
  if (!need(eda, t)) return;
  const g = eda!.benford_per_kelompok["semua"]!;
  const bars = benfordBars(g.first_digit);
  assert.equal(bars.length, 9); assert.deepEqual(bars.map((b) => b.digit), [1, 2, 3, 4, 5, 6, 7, 8, 9]);
  assert.ok(Math.abs(bars.reduce((s, b) => s + b.observed, 0) - 1) < 1e-9 && Math.abs(bars.reduce((s, b) => s + b.expected, 0) - 1) < 1e-9);
  assert.ok(Math.abs(bars[0]!.expected - Math.log10(2)) < 1e-9);                           // P(digit 1) menurut Benford
  assert.equal(benfordBars(g.last_digit).length, 10);
  const h = heatmapCells(eda!.korelasi);
  assert.equal(h.labels.length, h.values.length); assert.ok(h.values.every((row) => row.length === h.labels.length));
  assert.ok(h.values.every((row, i) => Math.abs((row[i] ?? NaN) - 1) < 1e-9));            // diagonal = 1
  const cls = classBars(eda!.distribusi_per_tahun!);
  assert.ok(cls.every((b) => b.lo! <= b.value && b.value <= b.hi!));
  assert.equal(cls.length, eda!.distribusi_per_tahun!.length);
});

test("Evaluasi: kurva ROC/PR, kalibrasi, DCA, metrik", (t) => {
  if (!need(ev, t)) return;
  const models = ["logreg", "random_forest", "hist_gb"];
  for (const s of [...rocSeries(ev!.kurva, models), ...prSeries(ev!.kurva, models)]) {
    assert.ok(s.points.length > 10);
    assert.ok(s.points.every(([x, y]) => x >= 0 && x <= 1 && y >= 0 && y <= 1), s.name);
  }
  const roc = rocSeries(ev!.kurva, ["random_forest"])[0]!.points;
  assert.deepEqual(roc[0], [0, 0]); assert.deepEqual(roc[roc.length - 1], [1, 1]);
  assert.deepEqual(rocSeries(ev!.kurva, ["tidak_ada"]), []);
  const key = Object.keys(ev!.kurva_kalibrasi)[0]!;
  const cal = calibrationPoints(ev!.kurva_kalibrasi[key]);
  assert.ok(cal.length > 0 && cal.every((c) => c.lo <= c.y && c.y <= c.hi)); assert.deepEqual(calibrationPoints(undefined), []);
  const dca = dcaSeries(ev!.decision_curve["random_forest"]);
  assert.equal(dca.length, 3); assert.equal(dca[0]!.points.length, dca[1]!.points.length); assert.ok(dca[2]!.points.every(([, y]) => y === 0));
  const pr = metricBars(ev!.metrik_uji_bootstrap, "pr_auc", models);
  assert.deepEqual(pr.map((b) => b.key), models); assert.ok(pr.every((b) => b.lo! <= b.value && b.value <= b.hi!));
});

test("Penjelasan: forest plot, SHAP, beeswarm, matriks kesepakatan", (t) => {
  if (!need(ex, t)) return;
  const forest = oddsForest(ex!.interpretasi_odds_ratio);
  assert.equal(forest.length, ex!.interpretasi_odds_ratio.length); assert.ok(forest.every((f) => f.lo <= f.or && f.or <= f.hi && f.lo > 0));
  const bars = shapBars(ex!.shap_global["random_forest"]!, { dsri: "DSRI" });
  assert.ok(bars.every((b, i) => i === 0 || bars[i - 1]!.value >= b.value));                // urut menurun
  assert.ok(Math.abs(bars.reduce((s, b) => s + b.value, 0) - 100) < 1e-6);                  // porsi berjumlah 100%
  assert.equal(bars.find((b) => b.key === "dsri")?.label, "DSRI");
  const order = ex!.shap_global["logit_mle"]!.map((r) => r.fitur);
  const sw = beeswarmByFeature(ex!.beeswarm["logit_mle"]!, order);
  assert.equal(sw.length, order.length); assert.ok(sw.every((s) => s.points.length > 0 && s.points.every((p) => p.fitur === s.fitur)));
  const m = agreementMatrix(ex!.kesepakatan_peringkat.spearman);
  assert.equal(m.labels.length, m.values.length); assert.ok(m.values.every((r, i) => Math.abs(r[i]! - 1) < 1e-9));
  assert.deepEqual(agreementMatrix([]), { labels: [], values: [] });
  assert.equal(symmetricLimit([-3, 2, NaN]), 3); assert.equal(symmetricLimit([]), 1); assert.equal(symmetricLimit([0]), 1);
});

import { percentileRanks } from "../src/lib/chartData.ts";
test("percentileRanks: rentang 0-1, seri dirata-rata, urutan terjaga", () => {
  assert.deepEqual(percentileRanks([10, 20, 30]), [0, 0.5, 1]);
  assert.deepEqual(percentileRanks([5, 5, 5]), [0.5, 0.5, 0.5]);
  assert.deepEqual(percentileRanks([3, 1, 2, 2]), [1, 0, 0.5, 0.5]);
  assert.deepEqual(percentileRanks([]), []); assert.deepEqual(percentileRanks([7]), [0.5]);
  const r = percentileRanks([0.3, 0.1, 0.9, 0.5]);
  assert.ok(r[1]! < r[0]! && r[0]! < r[3]! && r[3]! < r[2]!);
});
