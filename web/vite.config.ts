import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  // Proxying keeps the API same-origin, so the app never hardcodes a host and
  // api.py's CORS allowance is only a fallback for running Vite off-proxy.
  server: { proxy: { '/api': 'http://localhost:8000' } },
  // flag-icons names ~540 flags, most of them under Vite's 4 KB inline
  // threshold. Left on, they all land in the stylesheet as data URIs and every
  // visitor downloads the whole world; emitted as files, a page fetches the two
  // flags it shows. Nothing else here is small enough to want inlining.
  build: { assetsInlineLimit: 0 },
})
