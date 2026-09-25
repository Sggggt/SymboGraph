# 开发与测试

日常操作从仓库根目录开始。运行环境见 [infra](../infra/README.md)，架构和协议见[技术白皮书](technical-spec.md)。本地验收结果保存在 Git 忽略的工作记录中。

## 依赖

前端使用根 workspace 和 `package-lock.json`：

```powershell
npm ci
npm run dev --workspace web
```

API 的 `pyproject.toml` 声明 Python 3.11 以上，Docker 使用锁定的 Python 3.13 与 `uv.lock`。镜像内通过 uv 同步依赖，Worker 复用 API 镜像。纯单元开发可使用隔离 Python 环境，但不能从宿主脚本直接修改生产形态 PostgreSQL/Qdrant/Redis。

根 `.env` 与根 `settings.json` 是互不重叠的配置维护入口，已有文件不要覆盖。前者只放秘密、连接和服务启动参数，后者只放非秘密运行、检索与构建参数；私有配置和验收输入不进入源码或公共 fixture。

## 修改顺序

1. 核对当前入口、文档和已有测试，明确变更影响的协议/数据身份。
2. 算法或契约先更新白皮书对应位置，再同步代码、API/shared、前端及 scripts。
3. 运行受影响回归；涉及事务、外部状态或历史读取时补充对应集成验证。
4. 需要真实模型/资料验收时先冻结计划和 gold，说明写入影响与时限。
5. 汇总通过、失败、跳过和未执行，把必要结果写入本地验收记录，清理临时产物。

当前 serving 主链为 `intent_execution_retrieval_v1`；旧控制链只保留持久记录读取和兼容测试。新增行为不得重新接入检索反思、生成前模型充分性、结果驱动修正或在线奖励。Web 使用宿主原生 Node.js，后端使用现有 Compose project；最终迁移与 gold 证据见交接。

接口使用状态按 App Router 入口可达性判断：页面能到达的组件实际导入 `api.ts` 包装器才算产品在用。FastAPI 已注册、OpenAPI 可见、测试覆盖或孤立组件引用只说明接口存在。接口变更时同步核对前端可达图、`api.ts`、router、scripts 和外部兼容调用方。

## 测试命令

```powershell
docker exec course-kg-api python -m pytest tests
npm run typecheck --workspace web
npm run lint --workspace web
npm run test --workspace web
npm run build --workspace web
python scripts/check_repository_hygiene.py
git diff --check
```

Web 开发与生产构建目录分离：`next dev` 使用 `.next`，`next build`/`next start` 使用 `.next-production`。若修改此约定，必须验证运行中的开发页在 production build 后仍加载当前 CSS；不能让两种进程共写一个 distDir。

Windows 启动器修改还要检查两种调用方式：双击语义下 `.bat` 必须在成功和失败后保留终态信息并等待按键；脚本化调用通过直接执行 `.ps1`，或为当前进程设置 `SYMBOGRAPH_NO_PAUSE=1` 后调用 `.bat`，必须保留底层 PowerShell 退出码。窗口等待只属于批处理包装层，不能改变 Compose、Web PID、日志或服务生命周期。

后端默认排除 `fallback_compat` 和 `no_fallback_e2e` 标记。前者只用于显式兼容测试，后者需要真实依赖与模型；是否运行、跳过原因和数据范围均要写清楚。普通单元通过不能替代真实服务验收。

涉及 Docker/依赖或实际链路时，先检查 smoke 计划：

```powershell
python scripts/docker_smoke.py --base-url http://127.0.0.1:8000/api
```

确认目标与影响后增加 `--execute`。仅清理文档和缓存时，不自动发送模型或数据写入请求。

## 回归矩阵

| 范围 | 应检查的行为 |
|---|---|
| 解析/结构 | 控制字符、标题与正文清洗、span remap、PDF 原生表格、未确定表示、失败回滚 |
| 构建 | TPE 试验/选中候选身份、数值边界、内存/映射路径、RQ 唯一主链、前缀/贡献/压缩及 grounding |
| 生命周期 | 版本推进、取消边界、before-image、Worker fork、outbox、缓存发布和可重试补偿 |
| 规划 | 意图/执行策略分离、直接 plan 与目录→详情→plan 的闭合状态机、一次安全 schema 反馈重提、读取/反馈/最终计划的持久轨迹顺序及 SSE/轮询/历史一致、紧凑提示 schema 与完整执行校验等价、提示分项字符预算、完整目录排序/超限、过滤与图身份、越权键、额外模型调用数、三个根入口、空词面合法性、权重/通道矛盾、能力清单与硬预算 |
| 检索 | 纯向量与混合、独立候选提名、RQ 重构、BM25 统计与父域投影、RRF 通道首候选保留、逐父探索、图 path label、空词面灰区与去重 |
| 索引 | BM25 分词/df/长度/统计域、流式完整 posting hash 与物化快照等价、跨语言排序、篡改 fail-closed、版本发布、取消/删除、旧索引隔离与通道依赖缓存 |
| 来源范围 | 文档族与具体文档、all/any/交并、标题/编号/角色定位、歧义/截断/跨版本、原文跨度、版本内目标扫描与跨版本结果等价 |
| 原文与回答 | 确定性来源准入、完整实际包/真实输入预算、PDF 原生结构元信息的闭合公共投影、部分回答与有界不足、一次原生 GFM 生成、合法引用即时转序号胶囊、非法引用原样输出、可见流与持久答案逐字一致、完整答案跨度绑定实际 Context Package 来源、无完成后正文替换、无额外结果模型调用 |
| 来源绑定 | 包内 handle、原文/回答身份、范围与路径、事务提交/回滚、篡改和历史重放；旧来源失效不阻断历史阅读或新检索 |
| 直接回答 | 能力卡、零工具/引用、同会话复用、失败回检索、SSE/同步一致、新旧协议隔离 |
| 可观测性 | prepared/completed 区分、接纳后先保存用户问题、错误类型、阶段计时、append-only 可见流与最终回答逐字一致、引用列表在正文后独立提交、`first_response_ms`/`first_token_ms`、10 秒 SSE 传输保活、断线后按 run 恢复、失败/取消会话终态、敏感信息不泄露 |
| Web | 无独立 `/search` 路由；问答历史自动恢复；过期会话 404 不产生 unhandled rejection；首次模型/会话加载只显示旋转图标；流式轨迹区分已完成步骤与正在执行的 run 阶段、展示等待时长且不把 keep-alive 算作步骤；生成期间逐增量渲染并平滑跟随、终态停止滚动；v3 来源绑定不误报不可核验；生成提示要求 GFM/LaTeX；Markdown/KaTeX 不被普通文本截断；显式单行槽位溢出省略并在悬停一秒后显示全文；开发/生产 distDir 隔离；桌面与移动端无横向溢出 |
| 配置 | 根 `.env`/`settings.json` 唯一键归属、分别原子更新、组合身份、文件变更但 Redis 未广播时的热加载与模型桥同步、生命周期、Redis 广播、单例刷新、重建/容器重建待办 |
| 仓库 | 文档链接、所有 CLI 已说明、帮助无外部依赖、生成物/密钥被忽略 |

历史回归保留已有记录的可读性，不作为目标新链路的执行步骤。模块名字不决定删除范围，以实际引用和持久数据依赖核对。

## 真实验收

测试前从原文确定问题、必要事实、来源范围和可接受的拒答/澄清。被测模型的回答不能反过来生成 gold；训练/开发样例与固定验收题分开。

冷构建统计从任务接纳到最终提交、缓存失效和 freshness 的总墙钟。QA 统计从接纳到最终回答/失败，记录意图/策略规划、来源定位、各通道提名与融合、分层图检索、装包、来源准入、一次生成和绑定。共享准备与嵌套阶段不能重复相加。

固定意图映射不能代替 LLM 策略验收；同一意图允许选择不同的合法层。按相同候选/时间预算比较单通道、固定权重与 LLM 权重，记录独立来源/主题覆盖和答案结果，暂不定义综合优化目标。

技术失败、语义错误、有效拒答、部分回答、未运行和成功事实答案分列。分位数使用 nearest-rank 并注明样本数；小样本不作稳定尾延迟结论。功能门禁和性能参考分别报告，不能用改口径覆盖旧失败。

direct-answer 行为改变时，在指定主库进行至少十次真实对话覆盖，含能力卡、来源复用、跨主题回检索和无依据请求。已有成功案例无理由不重跑；修复后的必要补验说明它与原轮是否同代码、同配置。

## 产物与清理

正式测试在 `apps/api/tests` 和 Web 对应测试文件，持久运维入口在 `scripts/`。一次性脚本、采样、镜像源码副本、截图和临时报告放 `output/`，核验后删除；必要结论写入 Git 忽略的本地验收记录。

私有冻结输入放在忽略的 `data/acceptance/`，不会随源码发布。默认 Compose 将仓库挂载为 `/workspace`，因此容器中可通过 `/workspace/data/acceptance/` 显式读取这些输入；这与应用原文卷 `/app/data` 不同。

清理前检查绝对路径和引用。保留 `.env`、依赖锁文件、当前运行镜像、数据卷和必要验收输入；不执行全局 Docker prune，不清理其他项目。
