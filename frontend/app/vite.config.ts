import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

const localApiTarget =
  process.env.VITE_PROXY_API_TARGET ||
  process.env.PAYFLOW_API_TARGET ||
  'http://127.0.0.1:8011'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 3006,
    proxy: {
      '/api': {
        target: localApiTarget,
        changeOrigin: true,
      },
    },
  },
})
