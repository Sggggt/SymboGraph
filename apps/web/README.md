# Web 前端

Next.js 16.2.4、React 和 TypeScript 提供资料库管理、上传、图谱、引用问答和设置界面。检索由问答流程调用，前端不再提供独立检索页。

## 代码地图

| 位置 | 内容 |
|---|---|
| `src/app/` | App Router 页面 |
| `src/components/app-shell.tsx` | 导航与资料库切换 |
| `upload-workspace.tsx` | 导入、全量重解析和批次状态 |
| `graph-panel.tsx` | 四层图与自然语言节点详情 |
| `qa-workspace.tsx` | 持久对话、SSE 终态、引用和受控诊断 |
| `markdown-renderer.tsx` | GFM、代码块与 KaTeX 富文本渲染 |
| `overflow-tooltip.tsx` | 卡片溢出检测与延迟一秒全文提示 |
| `settings-workspace.tsx` | Profile 与 Runtime Settings |
| `src/lib/api.ts` | API 请求和共享类型 |
| `src/lib/agent-trace.ts` | 状态和审计的展示映射 |

未写目录的组件文件位于 `src/components/`。

产品路由可达性以 App Router 入口、实际导入的组件和 `api.ts` 调用图为准。源码中存在组件或包装器不能作为产品在用证据。

## 开发

从仓库根安装锁定依赖：

```powershell
npm ci
npm run dev --workspace web
```

依赖提升后，本地 Next.js 文档通常在 `node_modules/next/dist/docs/`；也检查 `apps/web/node_modules/next/dist/docs/`。修改前以本地 16.2.4 文档和当前构建为准。

默认 API 为 `http://127.0.0.1:8000/api`。Web 已从本项目 Compose 移出，使用 `start-web.ps1` 或根 `start-app.ps1` 在宿主启动；资源管理器双击对应 `.bat` 时，启动窗口会在成功或失败后等待按键，服务继续在后台运行。端口、构建和运行方式见 [infra](../../infra/README.md)。不要创建另一份后端运行配置。

```powershell
npm run typecheck --workspace web
npm run lint --workspace web
npm run test --workspace web
```

## 界面边界

界面已取消普通/摘要模式选择器及请求中的 `retrieval_granularity`。意图、入口层和权重来自服务端 LLM 策略；产品页展示简短执行说明，具体分数/权重矩阵只放诊断入口。回答简洁度或格式偏好与检索层级分开，不能通过前端默认值重新建立固定映射。

服务端状态优先用 React Query，mutation 后明确失效缓存。会话消息使用稳定 query key 和无限 staleTime，同一 App 生命周期中离开再进入问答页直接显示缓存，不重新加载对话；完整刷新只用一次 messages 响应水合消息、引用和最近 trace，不逐 run 串行补取。进入问答页时自动恢复当前资料库最近的持久会话；用户显式新建会话后保持空白，`activeSessionId=null` 不等于存在水合任务。手动选择历史时先读取消息再提交 activeSessionId；删除时先乐观移除列表项，失败恢复缓存，成功后的服务端重取不延长 mutation。缓存项对应的服务端会话已不存在时移除该项、刷新列表并显示可读提示，不抛出未处理 Promise。模型配置和会话首次水合期间显示统一旋转图标，不把尚未返回的数据写成“模型不可用”，也不使用加载卡片或 skeleton。目标生成的 SSE token 直接追加到当前最终回答，结束后只接收引用列表和 final，不用 `answer_replace` 重写正文；该帧只为能力卡等没有 provider 正文流的兼容路由保留。生成期间跟随底部平滑滚动，终态后停止主动滚动。SSE 收到完成、失败或取消后，不能重新显示加载中；拿到 run id 后若连接中断，后台 owner 继续执行，页面保留本轮指针并轮询持久 run，恢复完成、失败或取消终态。SSE keep-alive 注释不显示为 trace。

全站内容使用可用宽度，不再由 `.kg-page` 固定居中上限压缩。圆角、完整边框和背景由组件自身决定；禁止全局把圆角改为 0 或只保留上下边框。问答恢复上一提交版的居中标题、嵌入式智能体消息和固定输入区，空状态不生成资料名或固定问题胶囊。运行设置按模型连接、运行控制、检索入口、图协议、重建参数和服务参数分页，桌面使用右侧嵌入导航，移动端在导航内部横向滚动。

主导航固定为概览、导入、问答、图谱和设置，`/search` 不属于产品路由。卡片及弹窗中需要单行收敛的标题/标识使用显式 `data-overflow-text` 或 `truncate` 槽位，超出边界时显示省略号，指针停留一秒后在可滚动提示层显示全文；普通段落保持自然换行，不能用全局叶子选择器强制单行。`.markdown-output` 继续由 ReactMarkdown、GFM、remark-math、rehype-katex 和 KaTeX CSS 处理段落、列表、表格、代码及行内/块公式。

开发服务器使用 `.next`，生产构建和 `next start` 使用 `.next-production`。两者分离后，可以在宿主开发服务运行时执行 production build，而不会把旧 CSS 或路由产物混入当前页面；两类目录均由 Git 忽略。

产品页展示回答、必要状态、短证据和原文引用。回答中的合法 `#source-n` 内部链接渲染为浅灰色序号胶囊，不执行页面跳转；错误引用格式保持普通原文。完整 plan、frontier、budget、UUID、hash 和原始 JSON 放在受控诊断入口。系统能力直答不显示虚假引用；历史证据复用展示本轮的新来源绑定。

引用卡片同时识别 `answer_source_binding_public_v1/v2/v3`。v3 以 `source_integrity_admission_hash` 和对应 observation id 作为唯一 authority，不与旧 verification 或 retrieval-gate authority 混用；合法 v3 引用显示“来源已核对”。

共享契约变动需要 typecheck 和相关组件/API 测试。本地验证结果保存在 Git 忽略的工作记录中。
