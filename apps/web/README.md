# SymboGraph Frontend Web App

A sleek, responsive, and visually premium Next.js client for exploring course Knowledge Graphs and interacting with the citation-grounded RAG query interface.

---

## 🚀 Technology Stack

This application is built with modern, production-grade frontend practices:

1. **Core Framework**: **Next.js 16.2.4** utilizing the App Router and **React 19** for optimized concurrency and state orchestration.
2. **State Management**: **TanStack Query (React Query) v5** handles server state synchronization, query caching, and mutation invalidation.
3. **Styling & UI Components**: Tailored HSL color palettes, modern typography, glassmorphism, dynamic transitions, and polished **Shadcn UI** primitives styled with Vanilla Tailwind CSS.
4. **Data Graph Visualization**: Interactive force-directed layouts representing course concepts, evidence relationships, topological centrality scores, and Louvain community grouping.

---

## 🎨 Core Views & Features

### 1. Interactive Knowledge Graph Explorer
- Fully interactive visual rendering of course concepts (nodes) and relations (edges).
- Highlights Dijkstra traversal paths, neighborhood relations, and Louvain community partitions.

### 2. Real-Time Ingestion Logs Monitor
- Connects to the backend log streamer using Server-Sent Events (SSE) / stream subscription.
- Visually shows document parsing, adaptive chunking distributions, embedding audits, and Auto HPO optuna trial evolution.

### 3. Citation-Grounded RAG Chat
- Sleek interactive chat dashboard with agentic reflection support.
- Displays inline hoverable citations mapping directly to source PDF bounding boxes, jupyter notebooks, or raw snippets.
- Reveals multi-hop reasoning steps and the retrieval layering breakdown.

### 4. Concept Card Catalog
- An inspector catalog showing concept definitions, Aliases, PageRank statistics, in/out degree counts, and all associated raw evidence snippets.

---

## 📂 Project Structure

```
apps/web/
├── src/
│   ├── app/            # Next.js App Router pages and layouts
│   ├── components/     # High-fidelity UI modular elements (graph, chat, cards)
│   ├── hooks/          # Custom react hooks for handling dynamic states
│   └── lib/
│       ├── api.ts      # Centralized, strongly-typed API client contracts
│       └── utils.ts    # Styling helpers and class-merging functions
├── public/             # Static graphics and icons assets
├── tsconfig.json       # TypeScript configuration aligning with build contracts
├── eslint.config.mjs   # Strict ESLint configurations
└── package.json        # Dependencies locked to Next.js 16.2.4
```

---

## 💻 Developer Scripts

Manage the frontend web app from the repository workspace using standard npm scripts:

### Starting Development Server
Starts a local development server at `http://localhost:3000`:
```bash
# Run from apps/web directory
npm run dev
```

### Static Production Build
Compiles the Next.js app for production deployment:
```bash
npm run build
```

### Strict Code Quality & Verification
To ensure backend compatibility, zero regressions, and robust builds, always run typechecking and linting before proposing changes:
```bash
# Run strict TypeScript type checks across the web workspace
npm run typecheck --workspace web

# Run ESLint validation
npm run lint --workspace web
```
