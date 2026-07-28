import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    // Dev only. In the container nginx proxies /api instead (with buffering
    // off, or SSE tokens arrive in one lump at the end).
    proxy: { "/api": "http://localhost:8000" },
  },
});
