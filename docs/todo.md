# SymboGraph 生产前待办（2026-08-21）

本清单用于清理后的最新源码本地开发栈，并为后续生产验收做准备。当前阶段明确要求源码挂载与热重载；不得把本轮镜像冒充不可变生产发布物。未验证项目不得标记完成；数据库、Qdrant、Redis 和 provider 操作只允许在 Docker Compose 内执行，模型与数据库 fallback 必须为 false。

## P0：Sample 小语料生产闭环（当前 RED）

历史 RED 基线：17 chunks、132/136 relation edges、52 Mid、20 Coarse、1286 Mid edges、189 Coarse edges、74 次概念 provider 请求；当时 QA 未生成 context package、answer 或 citation。后续摘要 QA 已生成有效 Context Package、answer 和 citation，但历史基线不能替代完整 parse → TPE → 四层图 → retrieval → QA 重建验收。

- [x] 将 TPE `edge_density` 升级为无向简单图归一化密度，并增加 scope-aware 稀疏边预算硬门；旧 `|E|/|V|` 仅保留历史诊断。
- [x] 小语料 TPE 的 K/reverse/bridge quota 采样范围按当前 chunk scope 收缩，保证 trial 域中存在可通过稀疏硬门的候选；不得沿用 `k_min=16` 生成近完全图。
- [x] 分离 RQ fuzzy routing 与 deterministic concept eligibility；完整 membership 保留，LLM eligibility 调用数为 0，Mid≤chunks、Coarse≤Mid，并报告语义压缩率。
- [x] 概念 provider 只处理 eligible packet；显著降低请求次数，完整成本证据写入 `output/`。
- [x] 修复 retrieval → context package → Agent answer → citation verification，消除无进展重复 replan，闭合取消/补偿状态；首条 golden case 为“DF数据集成指的是什么”。
- [x] 修复 upload 重解析用物理 source-slot hash 覆盖 Document title 的回归；文件列表、目录树、Search、Context Package 与 Citation 必须统一显示原始逻辑文件名，禁止 `本地文件`/`本地资料` 占位。
- [x] 为 Mid/Coarse provider output 与 semantic reuse 增加 natural-label hard gate，拒绝 `未命名概念`、RQ/Chunk/Prefix 地址标签、UUID/hash 等非业务概念名；fallback=false 重建后 active 概念必须全部具名。
- [x] QA/图谱页按线上 v6.2.2 恢复产品视觉与交互布局，同时继续禁止默认 UUID、hash、protocol、raw JSON、代码日志、完整 trace 和 conversation raw state；Search 清理错误来源文案。
- [x] 重跑 Graph/Search/QA 产品面负向回归与 Playwright，验证原始文件名、自然概念名、v6.2.2 布局基线和主界面无内部审计泄漏；完整证据写入 `output/` 后才能标完成。证据：`output/sample-v4-natural-identity-concept-label-v622-ui-acceptance-20260820.md`。
- [x] 恢复 v6.2.2 QA 分层思考轨迹 UI：生成中实时展开、历史回答可折叠回放；只显示产品化步骤投影，禁止 UUID/hash/protocol/raw scores/chunk ids/provider 原文，并补 Vitest 与真实 Playwright QA 证据。证据：`output/sample-v4-qa-v622-layered-trace-ui-acceptance-20260820.md`。
- [x] 修复设置页 `RQ softmax` 数字输入 step base（`0.35` 不再误报为仅允许 `0.340001/0.350001`）；修复摘要/Coarse QA 重复 executor path label 导致 citation provenance 误拒绝、空 repair 覆盖有效 package，以及完整 trace/citation 写入 localStorage 导致配额溢出。证据：`output/summary-mode-qa-and-settings-input-fix-20260821.md`。
- [x] 增加 `embedding_provider_probe_v1` 脱敏连通性脚本：默认 dry-run、`--execute` 才在 API 容器内走真实 EmbeddingProvider/model bridge；报告向量结构或 typed failure，不保存输入、向量、endpoint、凭据或 provider 原文，并补脚本回归与真实 probe 证据。Provider 臂复现 6 次失败，单次 bridge 臂直接返回 `502 upstream_transport_error`，从而排除 API/EmbeddingProvider 适配层；专项状态 RED。证据：`output/embedding-provider-ablation-20260820.md`。
- [x] 补后端、shared contract、前端负向与 golden-case 回归；gray-zone same-input same-decision 且模型调用数为 0。
- [x] 在 Docker Compose、`ENABLE_MODEL_FALLBACK=false`、`ENABLE_DATABASE_FALLBACK=false` 下重建 Sample，执行检索、Agent QA、citation、性能、成本和 Playwright UI 验收。
- [x] 将可复现 GREEN/RED 矩阵、命令、耗时、provider 计数、截图、日志与阻断项写入 `output/`；任何未验证项保持 RED。
- [x] 执行提交前隐私与仓库卫生审计：移除个人文件名、私有资料指纹和内部模型 endpoint，扩展 Git/Docker ignore，清理宿主临时文件、本地 SQLite 遗留、未使用 KnowledgeGraph 数据卷和旧镜像；真实配置只允许存在于仓库根 `.env`，证据写入 `output/pre-push-workspace-audit-20260821.md`。

## P0：源码挂载开发构建与运行边界

- [x] 将 Runtime Settings 收敛为仓库根 `.env` 单一真值；删除第二份 runtime env、runtime-config volume 和所有运行时 settings 文件副本。证据：`output/single-root-env-runtime-settings-acceptance-20260822.md`。
- [x] 前端、API、worker、beat、Compose、启动器统一绑定根 `.env`；PostgreSQL 清除全局 settings value snapshot，只保留 hash/changed keys/lifecycle/error audit。证据同上。
- [x] 设置页删除双值语义与相关学习成本，只展示已写入、已生效、待重建、待重启、失败；产品面不显示 runtime/bridge/candidate/profile hash、候选 UUID 或 raw metrics。证据同上。
- [x] 只读比较旧 runtime-config active/desired 与根 `.env`，迁移四个非密钥 Runtime Settings 差异并验证密钥 presence 一致后删除旧卷；hot/rebuild/service 三类生命周期均完成可逆探针，报告不记录参数值或密钥。证据同上。

- [x] API 镜像使用锁定 Python 基础镜像和 `uv.lock`，保留容器内 pytest/OCR 开发验收能力。
- [x] Web 镜像固定 Next.js 16.2.4，并保留 `next dev`、本地源码挂载和热更新能力。
- [x] Web 开发镜像包含仅用于跨语言 canonical-contract Vitest 的最小固定 Python 运行时；不得在产品 UI 路径调用 Python。
- [x] API、worker、beat、model bridge 和 Web 按 Compose 既有边界挂载源码；挂载必须只读，Web 源码挂载按 Next 开发服务器要求可写。
- [x] worker/beat 使用有界 `watchfiles` 包装直接 Celery 命令；API 使用 Uvicorn `--reload` + lifespan readiness，模型和数据库 fallback 均为 false。
- [x] 修复 Windows 启动/重建脚本的镜像引用边界：build 只接受显式 mutable tag，digest 只允许 `-SkipBuild` 运行；worker 复用 API 镜像，down 覆盖 model-bridge profile。证据：`output/start-app-launcher-fix-20260821.md`。
- [x] 修复完整设置表单的 hot apply 过滤：只把实际变化的 hot key 与显式 secret clear 送入 active apply，未变化 endpoint 不再触发模型桥 preflight/reload；恢复独立 RuntimeSettingsVersion publication advisory lock，修复 PostgreSQL 分支 `NameError`。证据：`output/runtime-settings-hot-apply-filter-fix-20260822.md`。
- [x] 根 `.env` writer 保持原子替换、跨进程锁、失败回滚和密钥脱敏；真实保存探针后无持久 sibling env/recovery/settings 文件。证据同上。
- [x] service-recreate 状态通过根 `.env` 与当前进程值比较得到；已删除二次“应用设置”脚本，值在根文件立即可见、当前 worker 池保持原值、恢复后 pending 清零。证据同上。
- [x] Docker 构建上下文排除 `output/`、依赖、缓存、数据和 ACL 阻塞临时目录。

## P0：构建与最小验证

- [x] `docker compose config` 通过，生产服务名保持 `course-kg-*`。
- [x] 构建最新 `course-kg-api:dev` 和 `course-kg-web:dev` 开发镜像。
- [x] API 镜像运行 Docker/镜像/Compose/readiness 相关回归；Web 通过 typecheck、lint、Vitest 和镜像 build。
- [x] 最终 API 容器全量后端回归通过：`1539 passed, 57 skipped`，共 1596 项；单一根 `.env`、迁移、启动器、UI 契约与错误分类均已纳入套件。
- [x] 启动全栈后 API、worker、beat、model bridge、PostgreSQL、Redis、Qdrant 健康，Web 返回 HTTP 200；所有业务容器 restart=0、OOM=false。
- [x] Web 使用容器内 Node HTTP healthcheck；运行栈不再包含 runtime bootstrap 服务。
- [x] 容器检查确认源码 bind mount 完整、热重载命令符合当前开发阶段，镜像 digest 与本次构建一致。
- [x] 执行三层热加载专项回归，并在运行容器中核对 API/worker/beat 的 `RUNTIME_ENV_FILE=/workspace/.env`、根文件 identity、Redis version 和 env sync 状态；未输出密钥值。证据同上。
- [x] fallback=false、模型协议/模型名和 secret presence 仅做脱敏核验，不记录密钥或 provider 原始响应。
- [ ] 当前 Docker smoke 使用“DF数据集成指的是什么”时，embedding provider probe GREEN，但 Search query perception 的 Anthropic chat 上游连续重试后仍返回 503；Search/QA/citation 本轮保持 RED。此前 GREEN 结果只作历史证据，不能替代本轮。证据：`output/docker_smoke_20260822_064842.json`、`output/probe_embedding_provider_20260822_064913.json`。

## P1：替换与清理

- [x] 保存旧 KnowledgeGraph 容器/镜像精确 inventory；排除 Dify 和所有其他 Compose 项目。
- [x] 新源码挂载开发栈验证 GREEN 后，删除被替代的旧 `course-kg-*` 容器。
- [x] 删除未被容器引用的旧业务镜像 `course-kg-web:local`；保留当前 `course-kg-api:dev`、`course-kg-web:dev` 及其基础镜像。
- [x] 单一根 `.env` 栈配置与测试验证 GREEN 后删除无容器引用的 KnowledgeGraph 旧 runtime-config volume；其他项目卷全部排除。删除前已把旧 desired Runtime Settings 差异迁移到根 `.env`。

## P1：交付证据

- [x] 将构建命令、退出码、镜像 digest/标签、容器健康、挂载、命令、资源和清理前后 inventory 写入 `output/production-build-20260820/`。
- [x] 输出最终 GREEN/RED 表；性能、依赖漏洞、动态 gray 覆盖或 provider 可用性若未验证，必须保持 RED。

## 尚未关闭的生产前阻断

- [ ] 修复或冻结接受 Alembic `check` 报告的模型/历史 schema index 与 nullable drift；当前数据库已在 head，但 autogenerate check 非空。
- [ ] Node 依赖已通过 ECharts 6.1 与非强制 lock 修复从 17 项降到 3 项 high；剩余均来自固定 Next.js 16.2.4 及其 PostCSS/Sharp，修复要求升级到 16.3.1。升级前必须同步 Next 版本约束、读取对应本地文档并完成全量 Web 回归，禁止 `npm audit fix --force`。
- [x] Python lock 已升级 h2、idna、LangChain/LangGraph、Pillow、Pydantic Settings、PyPDF、python-multipart、SoupSieve、Starlette 和 urllib3 等受影响依赖；新 API 镜像内 `pip-audit` 从 14 个包/62 条记录降为 0，并通过 locked build 与核心后端回归。
- [ ] 在当前 Sample 上重新运行完整 parse → TPE → 四层图 → retrieval → Agent QA → UI 验收；本轮已验证真实摘要 QA，但没有重新执行全量导入、构图、性能和成本矩阵。
- [ ] 重新执行性能、动态 gray coverage 和真实 provider 可用性/成本验收；本轮只验证 bridge、配置和 fallback 边界，没有发起 provider 请求。
- [x] 收敛 `tests/test_agent_repair_review.py` 中 3 个过时的 gray-path/carry-forward fixture 断言：稀疏小图允许没有 gray candidate，排除上一轮 Mid 后也允许 carry-only；若存在 gray decision 仍逐条校验冻结身份，no-progress gate 负责停止无新证据的 repair。
- [x] 删除 gap/independent-review/final-acceptance 开发阶段测试、旧 Sample terminal/fault/browser/preproduction 脚本及其测试；`scripts/README.md` 只保留当前通用运维入口，普通 checkout 不再依赖固定 Sample 验收归档。
- [x] 将 Alembic 历史压缩为最近一周的两段迁移：`20260820_0041` 是当前完整 schema baseline，`20260821_0042` 通过只读 target preflight 与显式 destructive flag 清理 4 张 Sample/first-import 开发表；新数据库可从零 `upgrade head`，旧 revision 与 migration snapshot 已删除。
- [x] 退役固定 Sample provider authorization、initial-activation/source-closure/handoff、first-import companion release 与 vector metadata migration 服务，以及对应 Celery retry/beat；首次构图复用通用 ingestion/context-graph 事务和恢复协议。
- [ ] 待推送的两个本地 commit 仍包含已从工作区删除的私有 endpoint/manifest 历史。由于禁止 reset/revert 和未获历史重写授权，本轮不得 push；必须先采用经用户明确授权的安全历史重写方案，并重新扫描 `origin/main..HEAD`。
