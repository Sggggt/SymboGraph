# SymboGraph Frontend Web App

A sleek, responsive, and visually premium Next.js client for exploring local knowledge graphs, configuring knowledge-base Profiles, and interacting with the citation-grounded RAG query interface.

---

## 🚀 Technology Stack

This application is built with modern, production-grade frontend practices:

1. **Core Framework**: **Next.js 16.2.4** utilizing the App Router and **React 19** for optimized concurrency and state orchestration.
2. **State Management**: **TanStack Query (React Query) v5** handles server state synchronization, query caching, and mutation invalidation.
3. **Styling & UI Components**: Tailored HSL color palettes, modern typography, glassmorphism, dynamic transitions, and polished **Shadcn UI** primitives styled with Vanilla Tailwind CSS.
4. **Data Graph Visualization**: Interactive force-directed layouts representing knowledge-base concepts, evidence relationships, topological centrality scores, and Louvain community grouping.
5. **Profile Configuration**: The Settings page includes a dedicated Profile tab with active binding status, copy/create/save/delete actions, AI-assisted Profile draft generation, advanced JSON diagnostics, and validation feedback.

---

## 🎨 Core Views & Features

### 1. Interactive Knowledge Graph Explorer
- Fully interactive visual rendering of course concepts (nodes) and relations (edges).
- Highlights Dijkstra traversal paths, neighborhood relations, and Louvain community partitions.

### 2. Real-Time Ingestion Logs Monitor
- Connects to the backend log streamer using Server-Sent Events (SSE) / stream subscription.
- Visually shows document parsing, adaptive chunking distributions, embedding audits, and Auto HPO optuna trial evolution.
- Distinguishes parsing, graph extraction, cancelling, and compensating phases so users can see whether cancellation rolled back parse writes or only restored graph state.

### 3. Citation-Grounded RAG Chat
- Sleek interactive chat dashboard with agentic reflection support.
- Displays inline hoverable citations mapping directly to source PDF bounding boxes, jupyter notebooks, or raw snippets.
- Reveals multi-hop reasoning steps and the retrieval layering breakdown.

### 4. Concept Card Catalog
- An inspector catalog showing concept definitions, Aliases, PageRank statistics, in/out degree counts, and all associated raw evidence snippets.

### 5. Runtime Settings & Full Reparse Controls
- Runtime settings writes model endpoints, graph budgets, retrieval toggles, and bounded concurrency controls through the backend `.env` update API.
- Upload controls expose selected-file parsing separately from full reparse. Full reparse is disabled until the course has active chunks, matching backend chunk-version rules.

### 6. Knowledge-Base Profile Settings
- New knowledge bases use the built-in default course Profile automatically.
- The built-in default Profile is read-only and cannot be deleted. Copy it before making custom changes.
- Any non-default Profile can be deleted. When a deleted Profile is still bound to one or more knowledge bases, the backend automatically rebinds those knowledge bases to the default Profile.
- The Profile Assistant opens as a streaming side panel. It returns natural-language guidance plus a JSON draft; users explicitly click **Autofill** and then **Save Profile** before the draft is persisted.
- The Advanced JSON editor shows live diagnostics with line/column, message, reason, and a red highlight on the first error line. The diagnostics pane and JSON editor are fixed-height scroll regions to avoid stretching the page.
- Creating a new knowledge base shows a visual workflow tutorial for env settings, Profile settings, and the import flow. The "do not show again" choice is stored in browser `localStorage` under `symbograph.hideCreateCourseTutorial`.

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

# Run API contract tests that include Profile endpoints
npm run test --workspace web -- src/lib/api.test.ts
```

Rendered changes to Profile dialogs, the advanced JSON editor, or the new-knowledge-base tutorial should be validated in a browser at `http://127.0.0.1:3000/settings` after the Docker web container has been rebuilt or its `.next` cache cleared.
