import { test } from "node:test";
import assert from "node:assert/strict";
import { divergingColor, extent, linePath, linearScale, niceDomain, niceTicks, percentileColor, quantile, sequentialColor, ticksIn, fitLabel } from "../src/lib/scales.ts";

test("linearScale memetakan dan menangani domain nol", () => {
  const s = linearScale(0, 10, 100, 200);
  assert.equal(s(0), 100); assert.equal(s(10), 200); assert.equal(s(5), 150); assert.equal(s(20), 300);
  assert.equal(linearScale(3, 3, 0, 100)(3), 50);
  assert.equal(linearScale(0, 1, 100, 0)(0.25), 75);                   // sumbu y terbalik
});

test("extent mengabaikan nilai tak hingga", () => {
  assert.deepEqual(extent([3, NaN, 1, null, undefined, 7, Infinity]), [1, 7]);
  assert.equal(extent([]), null); assert.equal(extent([NaN]), null);
});

test("niceTicks: kelipatan 1-2-5, mencakup rentang, tanpa galat floating-point", () => {
  assert.deepEqual(niceTicks(0, 1, 6), [0, 0.2, 0.4, 0.6, 0.8, 1]);
  assert.deepEqual(niceTicks(0, 100, 5), [0, 50, 100]);                    // langkah 25 dibulatkan ke 50 (aturan 1-2-5)
  assert.deepEqual(niceTicks(0, 100, 11), [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]);
  for (const [lo, hi, n] of [[0.0123, 0.987, 5], [-3.2, 8.7, 6], [0, 1e6, 5], [1e-5, 3e-5, 4], [-1, 1, 5], [37, 38, 5]] as const) {
    const t = niceTicks(lo, hi, n);
    assert.ok(t[0]! <= lo + 1e-12 && t[t.length - 1]! >= hi - 1e-12, `mencakup [${lo}, ${hi}]`);
    const steps = t.slice(1).map((v, i) => v - t[i]!);
    assert.ok(steps.every((s) => Math.abs(s - steps[0]!) < steps[0]! * 1e-6), "langkah seragam");
    assert.ok(t.every((v) => String(v).length < 14), "tanpa 0.30000000000000004");
  }
  assert.deepEqual(niceTicks(5, 5), [5]); assert.deepEqual(niceTicks(NaN, 1), []);
  assert.deepEqual(niceTicks(10, 0, 3), niceTicks(0, 10, 3));
  assert.deepEqual(niceDomain(0.02, 0.93, 5), [0, 1]);
  assert.deepEqual(niceDomain(NaN, 1), [0, 1]);
});

test("linePath memutus garis pada nilai tak terdefinisi", () => {
  assert.equal(linePath([[0, 0], [1, 1], [2, 2]]), "M0 0L1 1L2 2");
  assert.equal(linePath([[0, 0], [1, NaN], [2, 2], [3, 3]]), "M0 0M2 2L3 3");
  assert.equal(linePath([[null, 1]]), ""); assert.equal(linePath([]), "");
  assert.equal(linePath([[0.123456, 1.987654]]), "M0.12 1.99");
});

test("skala warna", () => {
  assert.equal(divergingColor(0), "rgb(255, 255, 255)");
  assert.equal(divergingColor(1), "rgb(215, 38, 61)");
  assert.equal(divergingColor(-1), "rgb(47, 111, 222)");
  assert.equal(divergingColor(5), divergingColor(1));                       // dipotong
  assert.equal(divergingColor(null), "rgb(240, 242, 247)");
  assert.equal(divergingColor(0.5, 0, 1), "rgb(255, 255, 255)");            // titik tengah domain kustom
  assert.equal(sequentialColor(0), "rgb(232, 240, 252)"); assert.equal(sequentialColor(1), "rgb(11, 31, 58)");
  assert.equal(sequentialColor(NaN), sequentialColor(0));
  assert.equal(percentileColor(0), "rgb(47, 111, 222)"); assert.equal(percentileColor(1), "rgb(215, 38, 61)");
});

test("quantile", () => {
  assert.equal(quantile([1, 2, 3, 4, 5], 0.5), 3); assert.equal(quantile([1, 2, 3, 4], 0.5), 2.5);
  assert.equal(quantile([3, 1, 2], 0), 1); assert.equal(quantile([3, 1, 2], 1), 3);
  assert.ok(Number.isNaN(quantile([], 0.5))); assert.equal(quantile([1, NaN, 3], 0.5), 2);
});

test("ticksIn tidak pernah melampaui domain eksplisit", () => {
  assert.deepEqual(ticksIn(1, 9, 6), [2, 4, 6, 8]);
  assert.deepEqual(ticksIn(0, 0.35, 5), [0, 0.1, 0.2, 0.3]);
  assert.deepEqual(ticksIn(0, 1, 6), [0, 0.2, 0.4, 0.6, 0.8, 1]);
  for (const [a, b] of [[0, 9], [0.02, 0.3], [-1, 1], [1, 9], [0.5, 0.52]] as const) {
    const t = ticksIn(a, b, 6);
    assert.ok(t.length > 0 && t.every((v) => v >= a - 1e-9 && v <= b + 1e-9), `[${a}, ${b}] -> ${t}`);
  }
  assert.deepEqual(ticksIn(9, 1, 6), ticksIn(1, 9, 6));
});

test("fitLabel memotong berdasarkan lebar dan tidak pernah melebihi batas", () => {
  assert.equal(fitLabel("Pendek", 200), "Pendek");
  const long = "Indeks Piutang terhadap Penjualan (DSRI)";
  const out = fitLabel(long, 200);
  assert.ok(out.endsWith("…") && out.length <= Math.floor(200 / 6.7) && out.length < long.length);
  assert.equal(fitLabel("abcdefghij", 10).length, 6);                  // batas minimum 6 karakter
  assert.equal(fitLabel("x".repeat(40), 1000), "x".repeat(40));
});
