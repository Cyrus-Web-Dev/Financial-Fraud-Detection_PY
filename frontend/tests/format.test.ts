import { test } from "node:test";
import assert from "node:assert/strict";
import { clamp, formatDeltaPoints, formatInt, formatNumber, formatPercent, formatProb, formatPValue, formatRupiahJuta, formatSmart, modelLabel, riskTone } from "../src/lib/format.ts";

test("angka bergaya Indonesia", () => {
  assert.equal(formatNumber(1234567.891, 2), "1.234.567,89");
  assert.equal(formatNumber(0.5, 3), "0,500");
  assert.equal(formatInt(1234.6), "1.235");
  assert.equal(formatPercent(0.0612), "6,1%");
  assert.equal(formatPercent(0.5, 0), "50%");
});

test("nilai tak terdefinisi menjadi tanda strip", () => {
  for (const f of [formatNumber, formatInt, formatPercent, formatProb, formatPValue, formatSmart, formatRupiahJuta, formatDeltaPoints]) {
    assert.equal(f(null), "–"); assert.equal(f(undefined), "–"); assert.equal(f(NaN), "–"); assert.equal(f(Infinity), "–");
  }
});

test("probabilitas dan p-value", () => {
  assert.equal(formatProb(0.0004), "< 0,1%");
  assert.equal(formatProb(0.005), "0,50%");
  assert.equal(formatProb(0.1234), "12,3%");
  assert.equal(formatProb(0), "0,00%");
  assert.equal(formatPValue(0.0004), "< 0,001");
  assert.equal(formatPValue(0.0527), "0,053");
});

test("rupiah juta dan format fleksibel", () => {
  assert.equal(formatRupiahJuta(1_500_000), "Rp 1,5 T");
  assert.equal(formatRupiahJuta(25_000), "Rp 25,0 M");
  assert.equal(formatRupiahJuta(850), "Rp 850 jt");
  assert.equal(formatRupiahJuta(-2_000_000), "-Rp 2,0 T");
  assert.equal(formatSmart(12345.6), "12.346");
  assert.equal(formatSmart(1.2345), "1,23");
  assert.equal(formatSmart(0.00012), "1,2e-4");
  assert.equal(formatSmart(0), "0");
});

test("selisih poin, clamp, tone, label model", () => {
  assert.equal(formatDeltaPoints(0.0123), "+1,2 poin");
  assert.equal(formatDeltaPoints(-0.05), "−5,0 poin");
  assert.equal(formatDeltaPoints(0), "0,0 poin");
  assert.equal(clamp(5, 0, 1), 1); assert.equal(clamp(-5, 0, 1), 0); assert.equal(clamp(0.3, 0, 1), 0.3);
  assert.deepEqual(["Tinggi", "Sedang", "Rendah", "x"].map(riskTone), ["high", "mid", "low", "low"]);
  assert.equal(modelLabel("random_forest"), "Random Forest"); assert.equal(modelLabel("tak_dikenal"), "tak_dikenal");
});
