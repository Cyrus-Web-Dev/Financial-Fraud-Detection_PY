import { test } from "node:test";
import assert from "node:assert/strict";
import { DEFAULT_PAGE, PAGES, groupedPages, hashFor, isPageId, pageDef, pageFromHash } from "../src/lib/router.ts";

test("hash <-> halaman", () => {
  assert.equal(pageFromHash("#/prediksi"), "prediksi"); assert.equal(pageFromHash("#prediksi"), "prediksi");
  assert.equal(pageFromHash("#/eda?x=1"), "eda"); assert.equal(pageFromHash("#/eda/sub"), "eda");
  for (const bad of ["", "#", "#/", "#/tidak-ada", "#/__proto__", "#//prediksi"]) assert.equal(pageFromHash(bad), DEFAULT_PAGE, bad);
  for (const p of PAGES) assert.equal(pageFromHash(hashFor(p.id)), p.id);
});

test("definisi halaman konsisten", () => {
  const ids = PAGES.map((p) => p.id);
  assert.equal(new Set(ids).size, ids.length);
  assert.ok(ids.every(isPageId) && !isPageId("x"));
  assert.equal(pageDef("eda").needsDataset, true); assert.equal(pageDef("beranda").needsDataset, false);
  const g = groupedPages();
  assert.deepEqual(g.map((x) => x.group), ["Mulai", "Data", "Model", "Sistem"]);
  assert.equal(g.flatMap((x) => x.pages).length, PAGES.length);
});
