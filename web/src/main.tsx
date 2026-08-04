import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

// Self-hosted through npm, never CDN-linked -- see CLAUDE.md. Solid is the only
// Font Awesome style used, so the other two stylesheets stay out of the bundle.
// ponytail: that still ships the whole 119 KB solid face for four glyphs. The
// prototype's pyftsubset step would cut it to ~2 KB -- do it if page weight
// starts to matter. Roboto is subsetted by unicode-range already, so a browser
// only fetches the latin slice.
import '@fontsource-variable/roboto'
import '@fontsource-variable/roboto-mono'
import '@fortawesome/fontawesome-free/css/fontawesome.css'
import '@fortawesome/fontawesome-free/css/solid.css'
// ponytail: this 28 KB stylesheet names every flag, so Vite emits all ~540 SVGs
// into dist. Only the two on screen are ever fetched. Narrow it if dist size
// ever matters -- transfer size is already fine.
import 'flag-icons/css/flag-icons.min.css'
import './styles.css'

import App from './App.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
