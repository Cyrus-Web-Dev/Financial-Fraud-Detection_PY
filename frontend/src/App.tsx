import { useEffect } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { useStore } from "./state/store.ts";
import type { PageId } from "./lib/router.ts";
import { Sidebar, TopBar } from "./components/layout/Shell.tsx";
import { ErrorBoundary } from "./components/layout/ErrorBoundary.tsx";
import { HomePage } from "./pages/HomePage.tsx";
import { UploadPage } from "./pages/UploadPage.tsx";
import { EdaPage } from "./pages/EdaPage.tsx";
import { FeaturesPage } from "./pages/FeaturesPage.tsx";
import { PredictPage } from "./pages/PredictPage.tsx";
import { CompanyPage } from "./pages/CompanyPage.tsx";
import { ExplainPage } from "./pages/ExplainPage.tsx";
import { EvaluationPage } from "./pages/EvaluationPage.tsx";
import { HistoryPage } from "./pages/HistoryPage.tsx";
import { ReportPage } from "./pages/ReportPage.tsx";
import { AboutPage } from "./pages/AboutPage.tsx";

function PageSwitch(props: { page: PageId }) {
  switch (props.page) {
    case "beranda": return <HomePage />;
    case "unggah": return <UploadPage />;
    case "eda": return <EdaPage />;
    case "fitur": return <FeaturesPage />;
    case "prediksi": return <PredictPage />;
    case "perusahaan": return <CompanyPage />;
    case "penjelasan": return <ExplainPage />;
    case "evaluasi": return <EvaluationPage />;
    case "riwayat": return <HistoryPage />;
    case "laporan": return <ReportPage />;
    case "tentang": return <AboutPage />;
  }
}

export default function App() {
  const page = useStore((s) => s.page);
  const open = useStore((s) => s.sidebarOpen);
  const toggle = useStore((s) => s.toggleSidebar);
  const init = useStore((s) => s.init);
  const sync = useStore((s) => s.syncPageFromHash);

  useEffect(() => {
    sync();
    void init();
    window.addEventListener("hashchange", sync);
    return () => window.removeEventListener("hashchange", sync);
  }, [init, sync]);

  useEffect(() => {
    window.scrollTo({ top: 0 });
  }, [page]);

  return (
    <div className="app">
      <Sidebar />
      <div className={`backdrop ${open ? "open" : ""}`} onClick={() => toggle(false)} />
      <main className="main">
        <TopBar />
        <div className="content">
          <ErrorBoundary key={page} label="Halaman ini">
            <AnimatePresence mode="wait">
              <motion.div key={page} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }}>
                <PageSwitch page={page} />
              </motion.div>
            </AnimatePresence>
          </ErrorBoundary>
        </div>
      </main>
    </div>
  );
}
