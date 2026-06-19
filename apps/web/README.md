# Web 前端

## 项目简介

`apps/web` 是 SymboGraph 的 Next.js 前端，用于管理本地知识库、上传资料、查看导入日志、浏览四层图谱、执行 layered search、展开 context package、进行引用问答并配置 profile/runtime settings。

## 目录

| 路径 | 职责 |
| --- | --- |
| `src/app/` | Next.js App Router 页面入口。 |
| `src/components/app-shell.tsx` | 应用壳层、导航、资料库切换。 |
| `src/components/overview-dashboard.tsx` | 知识库概览、导入状态和快捷入口。 |
| `src/components/upload-workspace.tsx` | 上传、全量重新解析、批次状态和中文日志流。 |
| `src/components/graph-panel.tsx` | Chunk Structure、Chunk Relations / RQ Membership、Mid Concepts、Coarse Concepts，以及双击节点后的自然语言详情卡片。 |
| `src/components/search-workspace.tsx` | Layered search、trace、context package 和图路径。 |
| `src/components/qa-workspace.tsx` | QA、Agent trace、citations、verification 和会话。 |
| `src/components/settings-workspace.tsx` | Profile settings 与 runtime settings。 |
| `src/lib/api.ts` | 后端 API 集中入口。 |
| `src/lib/agent-trace.ts` | Agent trace 中文展示映射。 |
| `src/lib/ingestion-log-meta.ts` | 导入日志阶段中文展示映射。 |

## 产品定位

Web 前端是四层图谱和问答审计的可视化入口。搜索页展示 deterministic layered traversal、RQ membership seed、frontier/path diagnostics 和 context package；QA 页面展示 Agent 计划、typed actions、observations、citation verification 和 repair；设置页分离 Profile Settings 与 Runtime Settings。

## 技术栈

| 范围 | 技术 |
| --- | --- |
| Framework | Next.js 16.2.4 App Router |
| UI | React 19.2.4, TypeScript, Tailwind CSS, lucide-react |
| 数据 | TanStack Query, shared TypeScript contracts |
| 图谱 | ECharts |
| 测试 | Vitest, ESLint, TypeScript typecheck |

修改 `apps/web` 前必须优先查看本地 Next.js 文档：

```text
apps/web/node_modules/next/dist/docs/
```

若本地文档不存在，以当前依赖版本、现有代码和实际构建结果为准。

## 主链路

```text
dashboard
-> upload / batch log
-> graph payload
-> layered search trace
-> RQ membership seed diagnostics
-> context package
-> QA stream
-> Agent trace groups
-> citation verification
-> runtime/profile settings
```

## 环境配置

Web 的浏览器端 API 地址必须使用宿主机可访问地址。Docker Compose 和 Dockerfile 当前都注入：

```env
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000/api
```

`infra/docker-compose.yml` 会按 `API_HOST_PORT` 生成 `http://127.0.0.1:${API_HOST_PORT:-8000}/api`；`src/lib/api.ts` 在未设置该变量时回退到 `http://localhost:8000/api`。

本地浏览器访问默认地址：

```text
http://127.0.0.1:3000
```

## 快速启动

推荐通过 Docker Compose：

```powershell
docker compose -f infra/docker-compose.yml up -d --build web
```

本地开发：

```powershell
npm run dev --workspace web
```

## 参数列表

| 分类 | 参数 |
| --- | --- |
| API | `NEXT_PUBLIC_API_BASE_URL` |
| Next.js | 版本固定在 `apps/web/package.json` 的 `next@16.2.4` |
| 测试 | `npm run typecheck --workspace web`, `npm run lint --workspace web`, `npm run test --workspace web` |

## 验证

从仓库根目录执行：

```powershell
npm run typecheck --workspace web
npm run lint --workspace web
npm run test --workspace web
```

前端可视变更需要浏览器检查关键页面：

```text
/upload
/search
/qa
/graph
/settings
```

## 运维测试

```powershell
python scripts/docker_smoke.py --base-url http://127.0.0.1:8000/api
```

截图、视觉 QA 和临时浏览器输出写入 `output/`，不提交。

## 文档

- [../../README.md](../../README.md)：仓库总览。
- [../../docs/technical-spec.md](../../docs/technical-spec.md)：技术白皮书。
- [../../packages/shared/src/index.ts](../../packages/shared/src/index.ts)：共享类型契约。

## 边界

- 前端 API 访问默认集中在 `src/lib/api.ts`。
- 服务端状态优先使用 TanStack Query，mutation 后明确失效相关 query。
- shared TS 类型必须与 Pydantic schema 和脚本输出同步。
- 用户可见 trace/log 文案中文优先，底层 JSON key 保持英文协议字段。
- 前端不保存 API key、Authorization header 或 provider 原始响应。
- 搜索页不触发完整 Agent P&E；QA 页面才展示 Agent plan/action/observation、verification 和 repair。
- 图谱页必须展示四层图，Mid 视为 RQ L3 投影，Coarse 视为 RQ L2 投影；右侧详情卡展示自然语言说明、关键数据、证据定位和相邻关系，不暴露旧版细粒度聚类、legacy graph 入口或 raw metadata JSON。
