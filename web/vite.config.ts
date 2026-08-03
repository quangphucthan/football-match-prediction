import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  // Proxying keeps the API same-origin, so the app never hardcodes a host and
  // api.py's CORS allowance is only a fallback for running Vite off-proxy.
  server: { proxy: { '/api': 'http://localhost:8000' } },
})
