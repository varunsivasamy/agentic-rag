import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/chat': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        timeout: 120000,       // 2 min — LLM calls can be slow
        proxyTimeout: 120000,
      },
      '/history': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/status': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
