import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "./",
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/health": "http://localhost:8000",
      "/policy": "http://localhost:8000",
      "/portfolio": "http://localhost:8000",
      "/models": "http://localhost:8000",
      "/tail-risk": "http://localhost:8000",
      "/bonus-malus": "http://localhost:8000"
    }
  }
});
