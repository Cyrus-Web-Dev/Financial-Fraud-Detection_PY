// Klien API bertipe. Menerima implementasi fetch (dapat disuntikkan untuk pengujian).
import type {
  ApiErrorBody, Catalog, EdaSummary, EvaluationResults, ExplanationResults, FeatureTable, Health, HistoryList, ModelDetail,
  ModelList, ModelingResults, PredictDatasetRequest, PredictRowsRequest, PredictionResult, Preview, ResultKind, UploadResult,
  ReportRequest, WhatIfRequest, WhatIfResult,
} from "./types.ts";

export class ApiError extends Error {
  kode: string;
  status: number;
  rincian: unknown;
  constructor(kode: string, pesan: string, status: number, rincian?: unknown) {
    super(pesan);
    this.name = "ApiError";
    this.kode = kode;
    this.status = status;
    this.rincian = rincian;
  }
}

export type Fetcher = (input: string, init?: RequestInit) => Promise<Response>;
export type ApiOptions = { baseUrl?: string; fetchImpl?: Fetcher; timeoutMs?: number };

function isErrorBody(x: unknown): x is ApiErrorBody {
  return typeof x === "object" && x !== null && "error" in x && typeof (x as ApiErrorBody).error?.kode === "string";
}

export function createApi(opts: ApiOptions = {}) {
  const base = (opts.baseUrl ?? "/api").replace(/\/+$/, "");
  const doFetch: Fetcher = opts.fetchImpl ?? ((input, init) => fetch(input, init));
  const defaultTimeout = opts.timeoutMs ?? 120_000;

  /** Kirim permintaan; gagal (jaringan/waktu habis/HTTP non-2xx) selalu menjadi ApiError berbahasa Indonesia. */
  async function send(method: string, path: string, init: { body?: BodyInit; json?: unknown; query?: Record<string, unknown>; timeoutMs?: number; accept?: string } = {}): Promise<Response> {
    const url = new URL(base + path, "http://x"); // basis semu: hanya untuk menyusun query string
    for (const [k, v] of Object.entries(init.query ?? {})) {
      if (v !== undefined && v !== null && v !== "") url.searchParams.set(k, String(v));
    }
    const finalUrl = base + path + (url.search ? url.search : "");
    const headers: Record<string, string> = { Accept: init.accept ?? "application/json" };
    let body = init.body;
    if (init.json !== undefined) {
      headers["Content-Type"] = "application/json";
      body = JSON.stringify(init.json);
    }
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), init.timeoutMs ?? defaultTimeout);
    let res: Response;
    try {
      const req: RequestInit = { method, headers, signal: ctrl.signal };
      if (body !== undefined) req.body = body;
      res = await doFetch(finalUrl, req);
    } catch (e) {
      const aborted = e instanceof Error && e.name === "AbortError";
      throw new ApiError(aborted ? "WAKTU_HABIS" : "JARINGAN", aborted ? "Permintaan terlalu lama. Coba lagi atau kurangi ukuran data." : "Tidak dapat terhubung ke server. Pastikan API berjalan.", 0);
    } finally {
      clearTimeout(timer);
    }
    if (!res.ok) {
      let data: unknown = null;
      try {
        const text = await res.text();
        data = text ? JSON.parse(text) : null;
      } catch {
        data = null;
      }
      if (isErrorBody(data)) throw new ApiError(data.error.kode, data.error.pesan, res.status, data.error.rincian);
      throw new ApiError("HTTP_" + res.status, `Server mengembalikan kesalahan (${res.status}).`, res.status);
    }
    return res;
  }

  async function request<T>(method: string, path: string, init: { body?: BodyInit; json?: unknown; query?: Record<string, unknown>; timeoutMs?: number } = {}): Promise<T> {
    const res = await send(method, path, init);
    const text = await res.text();
    let data: unknown = null;
    if (text) {
      try {
        data = JSON.parse(text);
      } catch {
        data = null;
      }
    }
    if (data === null) throw new ApiError("RESPON_TIDAK_VALID", "Respons server tidak dapat dibaca.", res.status);
    return data as T;
  }

  async function requestBlob(method: string, path: string, init: { json?: unknown; timeoutMs?: number } = {}): Promise<{ blob: Blob; filename: string }> {
    const res = await send(method, path, { ...init, accept: "application/pdf" });
    const blob = await res.blob();
    if (blob.size === 0) throw new ApiError("RESPON_TIDAK_VALID", "Server mengembalikan berkas kosong.", res.status);
    return { blob, filename: filenameFromDisposition(res.headers.get("Content-Disposition")) };
  }

  return {
    health: () => request<Health>("GET", "/health", { timeoutMs: 10_000 }),
    models: () => request<ModelList>("GET", "/models", { timeoutMs: 15_000 }),
    modelDetail: (name: string) => request<ModelDetail>("GET", `/models/${encodeURIComponent(name)}`),
    catalog: () => request<Catalog>("GET", "/features/catalog"),
    uploadDataset: (file: Blob, filename: string) => {
      const form = new FormData();
      form.append("file", file, filename);
      return request<UploadResult>("POST", "/datasets", { body: form });
    },
    loadSample: (kind: "bersih" | "kotor" = "bersih") => request<UploadResult>("POST", "/datasets/sample", { query: { kind } }),
    preview: (id: string, rows = 20, offset = 0) => request<Preview>("GET", `/datasets/${encodeURIComponent(id)}/preview`, { query: { rows, offset } }),
    eda: (id: string) => request<EdaSummary>("GET", `/datasets/${encodeURIComponent(id)}/eda`),
    features: (id: string, rows = 20) => request<FeatureTable>("GET", `/datasets/${encodeURIComponent(id)}/features`, { query: { rows } }),
    deleteDataset: (id: string) => request<{ dihapus: string }>("DELETE", `/datasets/${encodeURIComponent(id)}`),
    predictDataset: (req: PredictDatasetRequest) => request<PredictionResult>("POST", "/predict/dataset", { json: req }),
    predictRows: (req: PredictRowsRequest) => request<PredictionResult>("POST", "/predict/rows", { json: req }),
    whatIf: (req: WhatIfRequest) => request<WhatIfResult>("POST", "/what-if", { json: req }),
    results: <K extends ResultKind>(kind: K, section?: string) =>
      request<K extends "evaluation" ? EvaluationResults : K extends "explanation" ? ExplanationResults : K extends "modeling" ? ModelingResults : unknown>(
        "GET", `/results/${encodeURIComponent(kind)}`, { query: { section } }),
    history: (p: { limit?: number; offset?: number; model?: string; company_id?: string; only_flagged?: boolean } = {}) =>
      request<HistoryList>("GET", "/history", { query: { ...p, only_flagged: p.only_flagged ? "true" : undefined } }),
    clearHistory: () => request<{ dihapus: number }>("DELETE", "/history"),
    report: (req: ReportRequest) => requestBlob("POST", "/report", { json: req, timeoutMs: 180_000 }),
  };
}

export type Api = ReturnType<typeof createApi>;

/** Nama berkas dari header Content-Disposition; hanya karakter aman (tanpa path) dan bawaan 'laporan.pdf'. */
export function filenameFromDisposition(header: string | null | undefined): string {
  const m = header ? /filename\*?=(?:UTF-8'')?"?([^";]+)"?/i.exec(header) : null;
  const raw = m?.[1] ? decodeURIComponent(m[1]) : "";
  const clean = raw.split(/[\\/]/).pop()?.replace(/[^A-Za-z0-9._-]/g, "_") ?? "";
  return clean && /\.pdf$/i.test(clean) ? clean : "laporan.pdf";
}

export function errorMessage(e: unknown): string {
  if (e instanceof ApiError) return e.message;
  if (e instanceof Error) return e.message;
  return "Terjadi kesalahan tak terduga.";
}
