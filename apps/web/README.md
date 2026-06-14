# Web 前端

`apps/web` 是 SymboGraph 的 Next.js 前端，用于管理本地知识库、上传资料、查看导入日志、浏览四层图谱、执行 layered search、展开 context package、进行引用问答并配置 profile/runtime settings。

## 技术栈

```text
Next.js 16.2.4
React 19.2.4
TypeScript
TanStack Query
Tailwind CSS
ECharts
Vitest
ESLint
```

修改 `apps/web` 前，优先查看本地 Next.js 文档：

```text
apps/web/node_modules/next/dist/docs/
```

如果本地文档不存在，以当前依赖版本、现有代码和实际构建结果为准。

## 目录

| 路径 | 职责 |
| --- | --- |
| `src/app/` | App Router 页面入口。 |
| `src/components/app-shell.tsx` | 应用壳层和导航。 |
| `src/components/overview-dashboard.tsx` | 知识库概览。 |
| `src/components/upload-workspace.tsx` | 上传、导入和日志。 |
| `src/components/graph-panel.tsx` | Chunk Structure、Chunk Relations、Mid Concepts、Coarse Concepts。 |
| `src/components/search-workspace.tsx` | Layered search、score components、graph expansion steps。 |
| `src/components/qa-workspace.tsx` | Context package、citations、verification、Agent trace。 |
| `src/components/settings-workspace.tsx` | Profile 和 runtime settings。 |
| `src/lib/api.ts` | 后端 API 集中入口。 |
| `src/lib/agent-trace.ts` | Agent trace 展示转换。 |

## 启动

推荐通过 Docker Compose：

```powershell
docker compose -f infra/docker-compose.yml up -d --build web
```

访问：

```text
http://127.0.0.1:3000
```

本地开发：

```powershell
npm run dev --workspace web
```

## 验证

从仓库根目录执行：

```powershell
npm run typecheck --workspace web
npm run lint --workspace web
npm run test --workspace web
```

## 前端边界

- 服务端状态优先使用 TanStack Query。
- mutation 后必须明确失效相关 query。
- API 类型与后端 Pydantic schema、`packages/shared` 保持同步。
- 图谱页展示四层：Chunk Structure、Chunk Relations、Mid Concepts、Coarse Concepts。
- 图谱页展示 full counts、sampled counts、freshness、hash、stale reason、grounding 和 retrieval contribution。
- 搜索页走 layered search，展示 layered route、concept path、score components、graph expansion steps 和结构上下文。
- QA 页面展示 context package、前后文 chunk、结构路径、图扩展步骤、agent plan、typed actions、observations、budget usage、citations、verification 和 repair actions。
- 多轮 QA 页面展示 conversation state 中的 active user constraints、任务状态和过往 context package/answer session 引用。
- 前端不保存 API key、Authorization header 或 provider 原始响应。
