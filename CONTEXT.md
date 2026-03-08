# NeuroLife - Project Context

## Overview
NeuroLife is a rebranded version of SakanaAI/asal (Automated Search for Artificial Life with Foundation Models). It consists of two parts:

1. **Python/ML Research Repo** (root level) - JAX-based implementation for discovering artificial life simulations using vision-language foundation models (CLIP, DINOv2)
2. **Publication Website** (`website/` directory) - Next.js static site replicating the academic paper page from pub.sakana.ai/asal/

## Rebranding Rules
- "ASAL" → "NeuroLife" in all user-facing text (README, notebook markdown, website)
- Code-level identifiers (variable names, function names, file names like `asal_metrics.py`) are kept as-is
- Primary author link: https://x.com/leechase99
- GitHub: leechase99/neuro-life

## Structure
```
neuro-life/
├── README.md                    # Rebranded README
├── neurolife.ipynb              # Rebranded notebook (42 cells)
├── main_opt.py                  # Supervised target & open-endedness optimization
├── main_illuminate.py           # Illumination via genetic algorithm
├── main_sweep_gol.py            # Game of Life brute force search
├── rollout.py                   # Simulation rollout
├── asal_metrics.py              # Metrics (filename kept as code-level)
├── util.py                      # Utilities
├── requirements.txt             # Python deps (JAX, evosax, etc.)
├── LICENSE                      # Apache 2.0
├── substrates/                  # ALife substrate implementations (Lenia, Boids, etc.)
├── foundation_models/           # FM wrappers (CLIP, DINOv2, pixels)
├── website/                     # Next.js publication site
│   ├── src/app/page.tsx         # Main publication page with all content
│   ├── src/app/layout.tsx       # Root layout (Roboto font, metadata)
│   ├── src/app/globals.css      # Publication styling (Distill-like)
│   ├── src/components/
│   │   ├── LazyVideo.tsx        # IntersectionObserver-based lazy video
│   │   └── BibTeX.tsx           # Citation block with copy button
│   └── public/assets/           # Downloaded media from pub.sakana.ai
│       ├── mp4/                 # 8 video files
│       └── png/                 # Images + equations
└── CONTEXT.md                   # This file
```

## Tech Stack
- **Python**: JAX, evosax, CLIP, DINOv2, einops, matplotlib
- **Website**: Next.js 16 (App Router), TypeScript, Tailwind CSS, react-intersection-observer
- **Deployment**: Vercel (set Root Directory to `website/`), domain: neurolifeblog.com
- **Image Generation**: Google Gemini (gemini-2.5-flash-image) via @google/genai SDK
  - API Key: AIzaSyB4w0j_-Lm7siDpCmcYMzSEYAoIYmzWCXU

## Key Decisions
- Media assets downloaded directly from pub.sakana.ai (authentic look)
- Cover image generated via Gemini saved at website/public/assets/png/neurolife_cover.png
- No backend/DB needed - pure static site
- Images set to `unoptimized: true` in next.config.ts for simpler deployment

## Build & Run
```bash
cd website
npm install
npm run dev    # Dev server at localhost:3000
npm run build  # Production build
```

## Vercel Deployment
Set Root Directory to `website/` in Vercel project settings. No other configuration needed.
