// Store aplikasi (Zustand). Logika murni ada di logic.ts; di sini hanya orkestrasi state dan pemanggilan API.
import { create } from "zustand";
import { createApi, errorMessage } from "../lib/api.ts";
import type { Api } from "../lib/api.ts";
import { DEFAULT_PAGE, hashFor, pageFromHash } from "../lib/router.ts";
import type { PageId } from "../lib/router.ts";
import type {
  Catalog, EdaSummary, EvaluationResults, ExplanationResults, FeatureTable, Health, HistoryList, ModelDetail, ModelList,
  ModelName, ModelingResults, PredictionResult, Preview, UploadResult, ValidationReport,
} from "../lib/types.ts";

export type Async = { loading: boolean; error: string | null };
export type DatasetInfo = {
  id: string; nama: string; nRows: number; hasLabel: boolean; validation: ValidationReport; preview: UploadResult["pratinjau"];
};
export type ReportInfo = { url: string; filename: string; size: number; model: ModelName; tahun: number | null };
export type AsyncKey =
  | "init" | "upload" | "eda" | "features" | "predict" | "evaluation" | "explanation" | "modeling" | "history" | "company" | "detail" | "report";

const IDLE: Async = { loading: false, error: null };

export type AppState = {
  api: Api;
  // navigasi
  page: PageId;
  sidebarOpen: boolean;
  setPage: (id: PageId) => void;
  syncPageFromHash: () => void;
  toggleSidebar: (open?: boolean) => void;
  // status asinkron per operasi
  status: Record<AsyncKey, Async>;
  // sistem
  health: Health | null;
  models: ModelList | null;
  catalog: Catalog | null;
  modelDetails: Partial<Record<ModelName, ModelDetail>>;
  init: () => Promise<void>;
  loadModelDetail: (name: ModelName) => Promise<ModelDetail | null>;
  // dataset
  dataset: DatasetInfo | null;
  eda: EdaSummary | null;
  featureTable: FeatureTable | null;
  companyData: { preview: Preview; features: FeatureTable } | null;
  uploadFile: (file: File) => Promise<boolean>;
  loadSample: (kind: "bersih" | "kotor") => Promise<boolean>;
  clearDataset: () => void;
  fetchEda: () => Promise<void>;
  fetchFeatures: () => Promise<void>;
  fetchCompanyData: () => Promise<void>;
  // prediksi
  model: ModelName;
  threshold: number | null;
  tahun: number | null;
  explain: boolean;
  prediction: PredictionResult | null;
  setModel: (m: ModelName) => void;
  setThreshold: (t: number | null) => void;
  setTahun: (y: number | null) => void;
  setExplain: (v: boolean) => void;
  runPrediction: () => Promise<boolean>;
  // laporan PDF
  report: ReportInfo | null;
  generateReport: (topN: number) => Promise<boolean>;
  clearReport: () => void;
  // hasil analisis tahap sebelumnya
  evaluation: EvaluationResults | null;
  explanation: ExplanationResults | null;
  modeling: ModelingResults | null;
  ensureEvaluation: () => Promise<void>;
  ensureExplanation: () => Promise<void>;
  ensureModeling: () => Promise<void>;
  // riwayat
  history: HistoryList | null;
  fetchHistory: (p?: { limit?: number; offset?: number; model?: string; only_flagged?: boolean }) => Promise<void>;
  clearHistory: () => Promise<void>;
};

const initialStatus = (): Record<AsyncKey, Async> => ({
  init: IDLE, upload: IDLE, eda: IDLE, features: IDLE, predict: IDLE, evaluation: IDLE, explanation: IDLE, modeling: IDLE,
  history: IDLE, company: IDLE, detail: IDLE, report: IDLE,
});

export function createAppStore(api: Api) {
  return create<AppState>()((set, get) => {
    /** Jalankan operasi dengan status loading/error terpisah per kunci. Mengembalikan undefined bila gagal. */
    async function run<T>(key: AsyncKey, fn: () => Promise<T>): Promise<T | undefined> {
      set((s) => ({ status: { ...s.status, [key]: { loading: true, error: null } } }));
      try {
        const out = await fn();
        set((s) => ({ status: { ...s.status, [key]: IDLE } }));
        return out;
      } catch (e) {
        set((s) => ({ status: { ...s.status, [key]: { loading: false, error: errorMessage(e) } } }));
        return undefined;
      }
    }

    function revokeReport(): void {
      const r = get().report;
      if (r && typeof URL !== "undefined") URL.revokeObjectURL(r.url);
    }

    function applyDataset(r: UploadResult): void {
      revokeReport();
      set({
        report: null,
        dataset: { id: r.dataset_id, nama: r.nama, nRows: r.n_baris, hasLabel: r.punya_label, validation: r.validasi, preview: r.pratinjau },
        eda: null, featureTable: null, companyData: null, prediction: null, tahun: null,
      });
    }

    return {
      api,
      page: DEFAULT_PAGE,
      sidebarOpen: false,
      setPage: (id) => {
        set({ page: id, sidebarOpen: false });
        if (typeof window !== "undefined" && window.location.hash !== hashFor(id)) window.location.hash = hashFor(id);
      },
      syncPageFromHash: () => {
        if (typeof window === "undefined") return;
        const id = pageFromHash(window.location.hash);
        if (id !== get().page) set({ page: id, sidebarOpen: false });
      },
      toggleSidebar: (open) => set((s) => ({ sidebarOpen: open ?? !s.sidebarOpen })),

      status: initialStatus(),
      health: null,
      models: null,
      catalog: null,
      modelDetails: {},
      init: async () => {
        await run("init", async () => {
          const [health, models] = await Promise.all([api.health(), api.models()]);
          set({ health, models });
          const first = models.model.find((m) => m.tersedia);
          if (first && !models.model.some((m) => m.nama === get().model && m.tersedia)) set({ model: first.nama });
          try {
            set({ catalog: await api.catalog() });
          } catch {
            /* katalog opsional saat model belum siap */
          }
        });
      },
      loadModelDetail: async (name) => {
        const cached = get().modelDetails[name];
        if (cached) return cached;
        const d = await run("detail", () => api.modelDetail(name));
        if (d) set((s) => ({ modelDetails: { ...s.modelDetails, [name]: d } }));
        return d ?? null;
      },

      dataset: null,
      eda: null,
      featureTable: null,
      companyData: null,
      uploadFile: async (file) => {
        const r = await run("upload", () => api.uploadDataset(file, file.name));
        if (r) applyDataset(r);
        return !!r;
      },
      loadSample: async (kind) => {
        const r = await run("upload", () => api.loadSample(kind));
        if (r) applyDataset(r);
        return !!r;
      },
      clearDataset: () => {
        revokeReport();
        const d = get().dataset;
        if (d) void api.deleteDataset(d.id).catch(() => undefined);
        set({ dataset: null, eda: null, featureTable: null, companyData: null, prediction: null, tahun: null, report: null });
      },
      fetchEda: async () => {
        const d = get().dataset;
        if (!d || get().eda) return;
        const eda = await run("eda", () => api.eda(d.id));
        if (eda) set({ eda });
      },
      fetchFeatures: async () => {
        const d = get().dataset;
        if (!d || get().featureTable) return;
        const ft = await run("features", () => api.features(d.id, 20));
        if (ft) set({ featureTable: ft });
      },
      fetchCompanyData: async () => {
        const d = get().dataset;
        if (!d || get().companyData) return;
        const r = await run("company", async () => {
          const [preview, features] = await Promise.all([api.preview(d.id, 200, 0), api.features(d.id, 200)]);
          return { preview, features };
        });
        if (r) set({ companyData: r });
      },

      model: "random_forest",
      threshold: null,
      tahun: null,
      explain: false,
      prediction: null,
      setModel: (m) => set({ model: m, prediction: null }),
      setThreshold: (t) => set({ threshold: t }),
      setTahun: (y) => set({ tahun: y, prediction: null }),
      setExplain: (v) => set({ explain: v }),
      runPrediction: async () => {
        const d = get().dataset;
        if (!d) return false;
        const { model, threshold, tahun, explain } = get();
        const r = await run("predict", () =>
          api.predictDataset({ dataset_id: d.id, model, threshold, explain, tahun }));
        if (r) set({ prediction: r });
        return !!r;
      },

      report: null,
      generateReport: async (topN) => {
        const d = get().dataset;
        if (!d) return false;
        const { model, threshold, tahun } = get();
        const out = await run("report", () => api.report({ dataset_id: d.id, model, threshold, tahun, top_n: topN }));
        if (!out) return false;
        revokeReport();
        set({ report: { url: URL.createObjectURL(out.blob), filename: out.filename, size: out.blob.size, model, tahun } });
        return true;
      },
      clearReport: () => {
        revokeReport();
        set({ report: null });
      },

      evaluation: null,
      explanation: null,
      modeling: null,
      ensureEvaluation: async () => {
        if (get().evaluation) return;
        const r = await run("evaluation", () => api.results("evaluation"));
        if (r) set({ evaluation: r as EvaluationResults });
      },
      ensureExplanation: async () => {
        if (get().explanation) return;
        const r = await run("explanation", () => api.results("explanation"));
        if (r) set({ explanation: r as ExplanationResults });
      },
      ensureModeling: async () => {
        if (get().modeling) return;
        const r = await run("modeling", () => api.results("modeling"));
        if (r) set({ modeling: r as ModelingResults });
      },

      history: null,
      fetchHistory: async (p = {}) => {
        const h = await run("history", () => api.history({ limit: 100, ...p }));
        if (h) set({ history: h });
      },
      clearHistory: async () => {
        const r = await run("history", () => api.clearHistory());
        if (r) set({ history: { total: 0, data: [] } });
      },
    };
  });
}

export const useStore = createAppStore(createApi());
