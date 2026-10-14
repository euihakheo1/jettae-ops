/// <reference types="vitest/config" />
import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

// 제때옵스 웹 — Vite 설정
// - /api 는 개발 중 FastAPI로 프록시. 기본값은 `jettae serve`/`jettae demo`의 기본 포트 8765이고,
//   JETTAE_API_URL 환경 변수로 바꿀 수 있다(예: JETTAE_API_URL=http://127.0.0.1:9000 npm run dev).
// - 빌드 결과(dist)는 FastAPI가 정적 파일로 서빙
const apiTarget = process.env.JETTAE_API_URL ?? 'http://127.0.0.1:8765';
const proxy = {
  '/api': { target: apiTarget, changeOrigin: true },
  '/healthz': { target: apiTarget, changeOrigin: true },
};

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { port: 5173, proxy },
  preview: { port: 4173, proxy },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    sourcemap: false,
    chunkSizeWarningLimit: 2000,
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    css: false,
    include: ['src/**/*.test.{ts,tsx}'],
    testTimeout: 20000,
  },
});
