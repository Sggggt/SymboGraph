# API 测试

## 项目简介

`apps/api/tests` 覆盖 Four-Layer Context Graph RAG 的后端行为，重点验证导入、四层图谱、检索、QA、runtime settings、维护入口和 migration 兼容性。

## 目录

| 路径 | 职责 |
| --- | --- |
| `test_context_graph_pipeline.py` | fixed chunk、structure graph、relation/fine/RQ、mid/coarse、context package。 |
| `test_routes_and_maintenance.py` | API routes、runtime settings、maintenance 和 reconciliation。 |
| `test_*.py` | 其他后端回归测试。 |
| `README.md` | 本测试说明。 |

## 产品定位

API tests 是开发时的最小后端回归门禁。涉及 PostgreSQL、Qdrant、Redis、模型接口和 no-fallback 的真实集成路径仍以 Docker smoke、诊断脚本和真实语料验收补足。

## 技术栈

| 范围 | 技术 |
| --- | --- |
| Test runner | pytest |
| API contracts | Pydantic schemas |
| DB/model fixtures | SQLAlchemy models and local fixtures |
| Integration companion | scripts and Docker smoke |

## 主链路

测试覆盖：

```text
fixed token chunks
-> structure graph
-> contextual embedding/BM25 metadata
-> chunk relation/fine/RQ
-> mid/coarse graph states
-> layered retrieval trace
-> context package
-> QA/citation verification audit
-> runtime settings and maintenance
```

## 环境配置

从 `apps/api` 执行时使用本地测试配置；Docker 栈内执行时使用容器环境和 `/app/apps/api` 工作目录。

## 快速启动

从 `apps/api` 执行：

```powershell
python -m pytest tests
```

从仓库根目录的 Docker 栈执行：

```powershell
docker exec -w /app/apps/api course-kg-api python -m pytest tests
```

## 参数列表

| 分类 | 参数 |
| --- | --- |
| pytest | `-q`, `-k <pattern>`, `tests/<file>.py` |
| Docker | `course-kg-api`, workdir `/app/apps/api` |
| 报告 | 诊断和验收报告写入仓库根目录 `output/` |

## 验证

行为变更后按风险选择：

```powershell
python -m pytest tests
python -m pytest tests/test_context_graph_pipeline.py -q
python -m pytest tests/test_routes_and_maintenance.py -q
```

## 运维测试

与脚本验收配合：

```powershell
python scripts/check_technical_spec_compliance.py --knowledge-base-name 贝叶斯
python scripts/docker_smoke.py --base-url http://127.0.0.1:8000/api
```

## 文档

- [../README.md](../README.md)：API 后端说明。
- [../../../docs/technical-spec.md](../../../docs/technical-spec.md)：技术白皮书。
- [../../../scripts/README.md](../../../scripts/README.md)：运维脚本。

## 边界

- Bug fix 默认补回归测试。
- 数据、检索、embedding、四层图谱、概念图或导入链路变化时，同步维护脚本与 smoke。
- 测试 fixture、Pydantic schema、shared TS 类型和脚本输出必须保持契约一致。
- 单元测试不能替代 Docker 栈内的 no-fallback 集成验收。
