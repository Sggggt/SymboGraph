"use client";

import Link from "next/link";
import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { ArrowRight, Orbit, Radar, Sparkles, Zap } from "lucide-react";

import { fetchDashboard, fetchGraph } from "@/lib/api";
import { NetworkCanvas } from "@/components/network-canvas";
import { ErrorBlock, LoadingBlock } from "@/components/query-state";
import { useKnowledgeBaseContext } from "@/components/knowledge-base-context";

const batchStateLabels: Record<string, string> = {
  queued: "排队中",
  parsing: "解析中",
  chunking: "切块中",
  embedding: "向量化中",
  extracting_graph: "生成图谱中",
  cancel_requested: "正在取消",
  cancelling: "正在取消",
  compensating: "正在回滚",
  cancelled: "已取消",
  cancel_failed: "取消失败",
  completed: "已完成",
  partial_failed: "部分失败",
  failed: "失败",
  skipped: "已跳过",
  idle: "未启动",
};

const sourceTypeLabels: Record<string, string> = {
  pdf: "PDF",
  notebook: "笔记本",
  markdown: "Markdown 文档",
  text: "文本",
  image: "图片",
  docx: "Word 文档",
  pptx: "演示文稿",
  unknown: "未知",
};

function batchStateLabel(state?: string | null): string {
  return state ? batchStateLabels[state] ?? state : "未启动";
}

function sourceTypeLabel(type: string): string {
  return sourceTypeLabels[type] ?? type;
}

function GraphStaleBadge({ isStale }: { isStale?: boolean }) {
  if (!isStale) {
    return null;
  }
  return (
    <span title="该图谱已过期，建议重建图谱" className="rounded-full border border-rose-200/35 bg-rose-400/15 px-3 py-1 text-xs font-medium text-rose-50">
      已过期
    </span>
  );
}

export function OverviewDashboard() {
  const { selectedKnowledgeBaseId } = useKnowledgeBaseContext();
  const { data, isLoading, error } = useQuery({
    queryKey: ["dashboard", selectedKnowledgeBaseId],
    queryFn: () => fetchDashboard(selectedKnowledgeBaseId, { includeGraph: false }),
    enabled: Boolean(selectedKnowledgeBaseId),
  });
  const graphQuery = useQuery({
    queryKey: ["overview-graph", selectedKnowledgeBaseId],
    queryFn: () => fetchGraph(selectedKnowledgeBaseId, "chunk-relation", "overview"),
    enabled: Boolean(selectedKnowledgeBaseId),
  });

  const stats = useMemo(() => {
    if (!data) return [];
    const counts = (data.context_graph?.counts ?? {}) as Record<string, number>;
    return [
      { label: "文档原件", value: data.ingested_document_count },
      { label: "活跃片段", value: counts.active_chunks ?? data.chunk_count ?? 0 },
      { label: "关系边", value: counts.chunk_relation_edges ?? data.graph_relation_count ?? 0 },
      { label: "RQ 片段边", value: counts.rq_relation_edges ?? 0 },
      { label: "目录分组", value: data.tree.length },
    ];
  }, [data]);

  if (isLoading) {
    return <LoadingBlock rows={4} />;
  }
  if (error) {
    return <ErrorBlock message={(error as Error).message} />;
  }
  if (!data) {
    return null;
  }

  return (
    <div className="kg-page grid gap-6 xl:grid-rows-[auto_minmax(0,1fr)]">
      <section className="grid items-stretch gap-6 xl:grid-cols-[minmax(560px,0.9fr)_minmax(0,1.1fr)]">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-panel h-full min-w-0 rounded-[30px] p-5 lg:p-6"
        >
          <div className="space-y-5">
            <div className="max-w-4xl space-y-4">
              <p className="section-kicker">本地资料 / 知识智能</p>
              <h2 className="glow-text text-3xl font-semibold leading-tight text-white lg:text-4xl">本地知识图谱、向量检索与 RAG 联动。</h2>
              <p className="max-w-3xl text-sm leading-7 text-cyan-50/72">围绕本地原件、结构路径、片段关系、RQ-KMeans、概念图和上下文证据包展开。新资料导入后自动解析、固定切块、构建四层图谱并向量化。</p>
            </div>
            <div className="flex flex-wrap gap-3">
              <Link href="/upload" className="rounded-full border border-cyan-300/40 bg-cyan-300/12 px-4 py-2.5 text-xs uppercase tracking-[0.24em] text-white">
                开始导入
              </Link>
              <Link href="/graph" className="rounded-full border border-white/12 px-4 py-2.5 text-xs uppercase tracking-[0.24em] text-white/72 transition hover:text-white">
                打开图谱
              </Link>
            </div>
            <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
              {stats.map((stat) => (
                <div key={stat.label} className="metric-line rounded-[20px] border border-white/8 bg-white/[0.03] px-4 py-3">
                  <p className="text-[11px] uppercase tracking-[0.28em] text-white/45">{stat.label}</p>
                  <p className="mt-1.5 text-2xl font-semibold text-white">{stat.value}</p>
                </div>
              ))}
            </div>
          </div>
        </motion.div>

        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.08 }} className="glass-panel flex h-full min-w-0 flex-col rounded-[30px] p-3 lg:p-4">
          <div className="mb-2 flex items-center justify-between">
            <div>
              <p className="section-kicker">实时图谱</p>
              <div className="mt-1.5 flex flex-wrap items-center gap-2">
                <h3 className="text-xl font-semibold text-white">四层图谱热区</h3>
                <GraphStaleBadge isStale={graphQuery.data?.freshness?.is_stale} />
              </div>
            </div>
            <Orbit className="size-5 text-cyan-200" />
          </div>
          <div className="min-h-0 flex-1">
            {graphQuery.isLoading ? (
              <div className="flex h-[340px] items-center justify-center rounded-[24px] border border-white/8 bg-white/[0.02] text-sm text-white/55">
                图谱正在加载，概览数据已就绪
              </div>
            ) : graphQuery.error ? (
              <ErrorBlock message={(graphQuery.error as Error).message} />
            ) : (
              <NetworkCanvas graph={graphQuery.data ?? data.graph} height={340} />
            )}
          </div>
        </motion.div>
      </section>

      <section className="grid min-h-0 gap-6 lg:grid-cols-[0.8fr_1.2fr]">
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.14 }} className="glass-panel kg-scroll-panel rounded-[28px] p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="section-kicker">导入状态</p>
              <h3 className="mt-2 text-2xl font-semibold text-white">批次进度</h3>
            </div>
            <Zap className="size-5 text-cyan-200" />
          </div>
          <div className="mt-6 grid gap-4 sm:grid-cols-2">
            <div className="rounded-[24px] border border-white/8 bg-white/[0.03] p-5">
              <p className="text-xs uppercase tracking-[0.28em] text-white/45">最新批次</p>
              <p className="mt-3 text-2xl font-semibold text-white">{batchStateLabel(data.batch_status?.state)}</p>
              <p className="mt-2 text-sm text-white/55">
                {data.batch_status
                  ? `${data.batch_status.processed_files} / ${data.batch_status.total_files} 已处理`
                  : "尚未启动全量导入"}
              </p>
            </div>
            <div className="rounded-[24px] border border-white/8 bg-white/[0.03] p-5">
              <p className="text-xs uppercase tracking-[0.28em] text-white/45">向量模型模式</p>
              <p className="mt-3 text-2xl font-semibold text-white">{data.degraded_mode ? "降级不可用" : "真实模型链路"}</p>
              <p className="mt-2 text-sm text-white/55">{data.degraded_mode ? "当前未检测到真实模型链路" : "真实向量模型与四层图谱链路已启用"}</p>
            </div>
          </div>
          <div className="mt-6 space-y-3">
            {Object.entries(data.coverage_by_source_type).map(([type, count]) => (
              <div key={type} className="space-y-2">
                <div className="flex items-center justify-between text-sm text-white/68">
                  <span className="uppercase tracking-[0.2em]">{sourceTypeLabel(type)}</span>
                  <span>{count}</span>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-white/6">
                  <div className="h-full rounded-full bg-[linear-gradient(90deg,#61d9ff,#7b7cff)]" style={{ width: `${Math.min(100, count * 8)}%` }} />
                </div>
              </div>
            ))}
          </div>
        </motion.div>

        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }} className="grid min-h-0 gap-6 md:grid-cols-2">
          <div className="glass-panel kg-scroll-panel rounded-[28px] p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="section-kicker">资料目录</p>
                <h3 className="mt-2 text-2xl font-semibold text-white">目录脉络</h3>
              </div>
              <Radar className="size-5 text-cyan-200" />
            </div>
            <div className="mt-6 space-y-4">
              {data.tree.slice(0, 6).map((partition) => (
                <div key={partition.id} className="rounded-[22px] border border-white/8 bg-white/[0.03] px-5 py-4">
                  <div className="flex items-center justify-between gap-4">
                    <p className="text-base font-medium text-white">{partition.title}</p>
                    <span className="text-xs uppercase tracking-[0.25em] text-white/45">{partition.children?.length ?? 0} 个文档</span>
                  </div>
                  <div className="mt-3 flex flex-wrap gap-2">
                    {(partition.children ?? []).slice(0, 4).map((child) => (
                      <span key={child.id} className="rounded-full border border-white/10 px-3 py-1 text-xs text-cyan-50/74">
                        {child.title}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="glass-panel kg-scroll-panel rounded-[28px] p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="section-kicker">快捷入口</p>
                <h3 className="mt-2 text-2xl font-semibold text-white">高频入口</h3>
              </div>
              <Sparkles className="size-5 text-cyan-200" />
            </div>
            <div className="mt-6 space-y-4">
              {[
                { href: "/search", title: "搜索实验室", description: "带过滤条件的向量检索与目录联动视图。" },
                { href: "/qa", title: "问答实验室", description: "流式回答、证据轨迹和命中片段并行展开。" },
                { href: "/graph", title: "四层图谱", description: "查看片段结构图、片段关系图、中粒度概念图和粗粒度概念图。" },
              ].map((entry) => (
                <Link
                  key={entry.href}
                  href={entry.href}
                  className="group block rounded-[22px] border border-white/8 bg-white/[0.03] px-5 py-4 transition hover:border-cyan-300/35 hover:bg-cyan-300/[0.06]"
                >
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <p className="text-base font-medium text-white">{entry.title}</p>
                      <p className="mt-2 text-sm leading-6 text-white/58">{entry.description}</p>
                    </div>
                    <ArrowRight className="mt-1 size-4 text-cyan-100/70 transition group-hover:translate-x-1" />
                  </div>
                </Link>
              ))}
            </div>
          </div>
        </motion.div>
      </section>
    </div>
  );
}
