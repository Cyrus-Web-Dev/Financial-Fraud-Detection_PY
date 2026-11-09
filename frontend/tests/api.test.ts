import { test } from "node:test";
import assert from "node:assert/strict";
import { ApiError, createApi, errorMessage } from "../src/lib/api.ts";
import type { Fetcher } from "../src/lib/api.ts";
import { fixture } from "./helpers.ts";
import type { PredictionResult, UploadResult } from "../src/lib/types.ts";

type Call = { url: string; init: RequestInit };
function fake(handler: (url: string, init: RequestInit) => Response | Promise<Response>) {
  const calls: Call[] = [];
  const f: Fetcher = async (url, init) => { calls.push({ url, init: init ?? {} }); return handler(url, init ?? {}); };
  return { f, calls };
}
const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

test("GET menyusun URL, query, dan header dengan benar", async () => {
  const { f, calls } = fake(() => json(fixture("health")));
  const api = createApi({ baseUrl: "/api/", fetchImpl: f });
  assert.equal((await api.health()).status, "ok");
  await api.preview("a".repeat(32), 5, 10);
  await api.history({ limit: 3, model: "logreg", only_flagged: true });
  await api.history({ company_id: undefined, only_flagged: false });
  await api.results("evaluation", "biaya");
  await api.results("modeling");
  assert.deepEqual(calls.map((c) => c.url), [
    "/api/health", `/api/datasets/${"a".repeat(32)}/preview?rows=5&offset=10`, "/api/history?limit=3&model=logreg&only_flagged=true",
    "/api/history", "/api/results/evaluation?section=biaya", "/api/results/modeling"]);
  assert.equal(calls[0]!.init.method, "GET");
  assert.equal((calls[0]!.init.headers as Record<string, string>)["Accept"], "application/json");
});

test("POST JSON mengirim body dan Content-Type; fixture respons terbaca", async () => {
  const pred = fixture<PredictionResult>("predict_dataset");
  const { f, calls } = fake(() => json(pred));
  const api = createApi({ fetchImpl: f });
  const r = await api.predictDataset({ dataset_id: "x".repeat(32), model: "random_forest", threshold: 0.2, explain: false, tahun: 2023 });
  assert.equal(r.n_baris, pred.n_baris);
  assert.equal(calls[0]!.url, "/api/predict/dataset");
  assert.equal((calls[0]!.init.headers as Record<string, string>)["Content-Type"], "application/json");
  assert.deepEqual(JSON.parse(String(calls[0]!.init.body)), { dataset_id: "x".repeat(32), model: "random_forest", threshold: 0.2, explain: false, tahun: 2023 });
});

test("unggah memakai FormData dan tidak menetapkan Content-Type manual", async () => {
  const { f, calls } = fake(() => json(fixture<UploadResult>("upload_sample")));
  const api = createApi({ fetchImpl: f });
  const r = await api.uploadDataset(new Blob(["a,b\n1,2"], { type: "text/csv" }), "data.csv");
  assert.equal(r.n_baris, 3600);
  const init = calls[0]!.init;
  assert.ok(init.body instanceof FormData && (init.body as FormData).get("file") instanceof Blob);
  assert.equal((init.headers as Record<string, string>)["Content-Type"], undefined);   // boundary diisi otomatis oleh browser
  assert.equal(((init.body as FormData).get("file") as File).name, "data.csv");
});

test("error terstruktur dari server menjadi ApiError", async () => {
  const body = { error: { kode: "DATA_TIDAK_VALID", pesan: "Data tidak lolos validasi", rincian: { n_errors: 2 } } };
  const api = createApi({ fetchImpl: fake(() => json(body, 422)).f });
  await assert.rejects(api.models(), (e: unknown) => {
    assert.ok(e instanceof ApiError);
    assert.equal(e.kode, "DATA_TIDAK_VALID"); assert.equal(e.status, 422); assert.equal(e.message, "Data tidak lolos validasi");
    assert.deepEqual(e.rincian, { n_errors: 2 }); return true;
  });
});

test("respons non-JSON, kosong, dan 5xx ditangani", async () => {
  const cases: [Response, string][] = [
    [new Response("<html>Bad Gateway</html>", { status: 502 }), "HTTP_502"],
    [new Response("", { status: 500 }), "HTTP_500"],
    [new Response("bukan json", { status: 200 }), "RESPON_TIDAK_VALID"],
    [new Response("", { status: 200 }), "RESPON_TIDAK_VALID"],
    [json({ foo: "bar" }, 400), "HTTP_400"],
  ];
  for (const [resp, kode] of cases) {
    const api = createApi({ fetchImpl: fake(() => resp.clone()).f });
    await assert.rejects(api.health(), (e: unknown) => e instanceof ApiError && e.kode === kode, kode);
  }
});

test("kegagalan jaringan dan waktu habis", async () => {
  const down = createApi({ fetchImpl: async () => { throw new TypeError("fetch failed"); } });
  await assert.rejects(down.health(), (e: unknown) => e instanceof ApiError && e.kode === "JARINGAN" && /terhubung/.test(e.message));
  const slow = createApi({
    timeoutMs: 20,
    fetchImpl: (_u, init) => new Promise((_res, rej) => { init?.signal?.addEventListener("abort", () => { const e = new Error("abort"); e.name = "AbortError"; rej(e); }); }),
  });
  await assert.rejects(slow.models(), (e: unknown) => e instanceof ApiError && e.kode === "WAKTU_HABIS");
});

test("id dataset di-encode dan errorMessage", async () => {
  const { f, calls } = fake(() => json({ dihapus: "x" }));
  await createApi({ fetchImpl: f }).deleteDataset("../etc/passwd");
  assert.equal(calls[0]!.url, "/api/datasets/..%2Fetc%2Fpasswd"); assert.equal(calls[0]!.init.method, "DELETE");
  assert.equal(errorMessage(new ApiError("X", "pesan", 400)), "pesan"); assert.equal(errorMessage(new Error("e")), "e");
  assert.match(errorMessage(42), /tak terduga/);
});

import { filenameFromDisposition } from "../src/lib/api.ts";

test("laporan PDF: blob, nama berkas dari header, dan permintaan POST JSON", async () => {
  const bytes = new TextEncoder().encode("%PDF-1.4 isi");
  const { f, calls } = fake(() => new Response(bytes, { status: 200, headers: { "Content-Type": "application/pdf", "Content-Disposition": 'attachment; filename="laporan_fraud_logreg_20260101_1200.pdf"' } }));
  const api = createApi({ fetchImpl: f });
  const r = await api.report({ dataset_id: "a".repeat(32), model: "logreg", threshold: 0.2, tahun: 2023, top_n: 10 });
  assert.equal(r.filename, "laporan_fraud_logreg_20260101_1200.pdf");
  assert.equal(r.blob.size, bytes.length);
  assert.equal(await r.blob.text(), "%PDF-1.4 isi");
  assert.equal(calls[0]!.url, "/api/report"); assert.equal(calls[0]!.init.method, "POST");
  assert.equal((calls[0]!.init.headers as Record<string, string>)["Accept"], "application/pdf");
  assert.deepEqual(JSON.parse(String(calls[0]!.init.body)), { dataset_id: "a".repeat(32), model: "logreg", threshold: 0.2, tahun: 2023, top_n: 10 });
});

test("laporan PDF: galat JSON dari server dan berkas kosong menjadi ApiError", async () => {
  const body = { error: { kode: "TAHUN_TIDAK_ADA", pesan: "Tidak ada data untuk tahun 1999." } };
  const bad = createApi({ fetchImpl: fake(() => json(body, 404)).f });
  await assert.rejects(bad.report({ dataset_id: "a".repeat(32), model: "logreg" }), (e: unknown) => e instanceof ApiError && e.kode === "TAHUN_TIDAK_ADA" && e.status === 404);
  const empty = createApi({ fetchImpl: fake(() => new Response(new Uint8Array(0), { status: 200 })).f });
  await assert.rejects(empty.report({ dataset_id: "a".repeat(32), model: "logreg" }), (e: unknown) => e instanceof ApiError && e.kode === "RESPON_TIDAK_VALID");
  const down = createApi({ fetchImpl: async () => { throw new TypeError("x"); } });
  await assert.rejects(down.report({ dataset_id: "a".repeat(32), model: "logreg" }), (e: unknown) => e instanceof ApiError && e.kode === "JARINGAN");
});

test("filenameFromDisposition aman terhadap path traversal dan format tak lazim", () => {
  assert.equal(filenameFromDisposition('attachment; filename="a.pdf"'), "a.pdf");
  assert.equal(filenameFromDisposition("attachment; filename=b.pdf"), "b.pdf");
  assert.equal(filenameFromDisposition("attachment; filename*=UTF-8''c%20d.pdf"), "c_d.pdf");
  assert.equal(filenameFromDisposition('attachment; filename="../../etc/passwd.pdf"'), "passwd.pdf");
  assert.equal(filenameFromDisposition('attachment; filename="..\\\\win\\\\x.pdf"'), "x.pdf");
  assert.equal(filenameFromDisposition('attachment; filename="evil.exe"'), "laporan.pdf");        // bukan PDF -> bawaan
  for (const h of [null, undefined, "", "attachment", "inline"]) assert.equal(filenameFromDisposition(h), "laporan.pdf");
});
