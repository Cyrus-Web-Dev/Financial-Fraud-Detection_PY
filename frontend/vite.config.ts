import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Pengembangan: frontend di :5173 meneruskan /api ke FastAPI di :8000 (tanpa masalah CORS).
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": { target: "http://127.0.0.1:8000", changeOrigin: true } } },
  build: { outDir: "dist", sourcemap: false, chunkSizeWarningLimit: 900 },
});
