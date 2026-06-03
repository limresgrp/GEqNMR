import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  cacheDir: '/tmp/geqnmr-vite-cache',
  plugins: [react()],
  server: {
    watch: {
      usePolling: true, // Enables hot reload inside Docker
    },
  },
})
