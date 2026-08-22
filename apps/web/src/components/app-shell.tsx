"use client";

import { useState } from "react";
import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion } from "framer-motion";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { BookOpenText, BrainCircuit, FolderPlus, Home, RefreshCw, Search, Settings, Share2, Sparkles, TerminalSquare, Trash2, Upload } from "lucide-react";

import { AmbientCanvas } from "@/components/ambient-canvas";
import { useKnowledgeBaseContext } from "@/components/knowledge-base-context";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { deleteKnowledgeBase, refreshKnowledgeBase } from "@/lib/api";
import { cn } from "@/lib/utils";

const navigation = [
  { href: "/", label: "概览", caption: "首页", icon: Home },
  { href: "/upload", label: "导入", caption: "导入", icon: Upload },
  { href: "/search", label: "搜索", caption: "搜索", icon: Search },
  { href: "/qa", label: "问答", caption: "对话", icon: BrainCircuit },
  { href: "/graph", label: "图谱", caption: "图谱", icon: Share2 },
  { href: "/settings", label: "设置", caption: "模型", icon: Settings },
];

const CREATE_KB_TUTORIAL_STORAGE_KEY = "symbograph.hideCreateKnowledgeBaseTutorial";

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const queryClient = useQueryClient();
  const { knowledgeBases, selectedKnowledgeBase, selectedKnowledgeBaseId, setSelectedKnowledgeBaseId, createKnowledgeBaseSpace, isCreating } = useKnowledgeBaseContext();
  const [createOpen, setCreateOpen] = useState(false);
  const [knowledgeBaseToDelete, setKnowledgeBaseToDelete] = useState<{ id: string; name: string } | null>(null);
  const [deleteKnowledgeBaseResult, setDeleteKnowledgeBaseResult] = useState<string | null>(null);
  const [nextKnowledgeBaseName, setNextKnowledgeBaseName] = useState("");
  const [refreshDone, setRefreshDone] = useState(false);
  const [createTutorialOpen, setCreateTutorialOpen] = useState(false);
  const [hideCreateTutorialDraft, setHideCreateTutorialDraft] = useState(false);
  const refreshMutation = useMutation({
    mutationFn: () => refreshKnowledgeBase(selectedKnowledgeBaseId),
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["knowledgeBases"] }),
        queryClient.invalidateQueries({ queryKey: ["dashboard", selectedKnowledgeBaseId] }),
        queryClient.invalidateQueries({ queryKey: ["knowledgeBase-files", selectedKnowledgeBaseId] }),
        queryClient.invalidateQueries({ queryKey: ["graph", selectedKnowledgeBaseId] }),
        queryClient.invalidateQueries({ queryKey: ["partition-graph"] }),
        queryClient.invalidateQueries({ queryKey: ["sessions", selectedKnowledgeBaseId] }),
      ]);
      setRefreshDone(true);
      window.setTimeout(() => setRefreshDone(false), 1600);
    },
  });
  const deleteKnowledgeBaseMutation = useMutation({
    mutationFn: (knowledgeBaseId: string) => deleteKnowledgeBase(knowledgeBaseId),
    onSuccess: async (data, deletedKnowledgeBaseId) => {
      const stats = data.stats;
      setDeleteKnowledgeBaseResult(
        `资料库数据已删除：Qdrant 点 ${stats.qdrant_points ?? stats.deleted_vectors ?? 0} 条，向量记录 ${stats.vector_records ?? stats.deleted_vector_records ?? 0} 条，文档 ${stats.documents ?? stats.deleted_documents ?? 0} 份，片段 ${stats.chunks ?? stats.deleted_chunks ?? 0} 个。`,
      );
      const nextKnowledgeBase = knowledgeBases.find((knowledgeBase) => knowledgeBase.id !== deletedKnowledgeBaseId) ?? null;
      setSelectedKnowledgeBaseId(nextKnowledgeBase?.id ?? null);
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["knowledgeBases"] }),
        queryClient.invalidateQueries({ queryKey: ["dashboard"] }),
        queryClient.invalidateQueries({ queryKey: ["knowledgeBase-files"] }),
        queryClient.invalidateQueries({ queryKey: ["graph"] }),
        queryClient.invalidateQueries({ queryKey: ["partition-graph"] }),
        queryClient.invalidateQueries({ queryKey: ["sessions"] }),
      ]);
    },
  });

  return (
    <div className="kg-future-field relative min-h-screen overflow-x-hidden bg-[#030714] text-foreground">
      <AmbientCanvas />

      <aside className="fixed inset-y-0 left-0 z-40 hidden w-[76px] border-r border-white/7 bg-[rgba(3,7,20,0.55)] backdrop-blur-2xl lg:flex lg:flex-col lg:items-center lg:gap-7 lg:py-6">
        <Image
          src="/diagraph-logo.svg"
          alt="SymboGraph"
          width={44}
          height={44}
          className="size-11 rounded-2xl border border-white/10 shadow-[0_0_28px_rgba(255,255,255,0.12)]"
          priority
        />

        <nav className="flex flex-col gap-2">
          {navigation.map(({ href, label, caption, icon: Icon }) => {
            const active = pathname === href;
            return (
              <Link key={href} href={href} className="group relative">
                <motion.div
                  whileHover={{ x: 3, y: -1 }}
                  whileTap={{ scale: 0.97 }}
                  className={cn(
                    "relative grid size-11 place-items-center rounded-2xl border border-transparent text-white/48 transition duration-200",
                    active && "border-cyan-300/25 bg-cyan-300/[0.075] text-white shadow-[0_0_24px_rgba(85,215,255,0.08)]",
                  )}
                >
                  <span
                    className={cn(
                      "absolute inset-y-2 -left-[17px] w-px rounded-full bg-transparent transition",
                      active && "bg-cyan-200/90 shadow-[0_0_12px_rgba(132,221,255,0.35)]",
                    )}
                  />
                  <Icon className="size-5 shrink-0" />
                </motion.div>
                <span className="pointer-events-none absolute left-full top-1/2 ml-3 -translate-y-1/2 rounded-full border border-white/8 bg-[rgba(5,9,24,0.92)] px-3 py-1 text-xs text-white/72 opacity-0 transition group-hover:opacity-100">
                  {caption} / {label}
                </span>
              </Link>
            );
          })}
        </nav>
      </aside>

      <div className="relative min-h-screen lg:pl-[76px]">
        <header className="fixed inset-x-0 top-0 z-30 border-b border-white/6 bg-[rgba(3,7,20,0.78)] backdrop-blur-2xl lg:left-[76px]">
          <div className="flex flex-wrap items-center justify-between gap-4 px-5 py-3 lg:px-7">
            <div className="flex min-w-0 items-center gap-3">
              <Image
                src="/diagraph-logo.svg"
                alt="SymboGraph"
                width={44}
                height={44}
                className="size-11 shrink-0 rounded-2xl border border-white/10 shadow-[0_0_28px_rgba(255,255,255,0.1)] lg:hidden"
                priority
              />
              <div className="min-w-0">
                <p className="text-[10px] uppercase tracking-[0.34em] text-cyan-100/42">知 识 库</p>
                <h1 className="mt-1 break-words text-lg font-semibold text-white lg:text-xl">本地资料智能检索台</h1>
                <p className="mt-1 text-xs text-white/45">{selectedKnowledgeBase?.name ?? "选择资料库空间"}</p>
              </div>
            </div>
            <div className="flex w-full min-w-0 flex-wrap items-center gap-2 lg:w-auto lg:justify-end">
              <div className="relative">
                <Button
                  type="button"
                  variant="outline"
                  size="icon"
                  className="rounded-full"
                  aria-label="刷新当前资料库"
                  onClick={() => refreshMutation.mutate()}
                  disabled={!selectedKnowledgeBaseId || refreshMutation.isPending}
                >
                  <RefreshCw className={cn("size-4", refreshMutation.isPending && "animate-spin")} />
                </Button>
                {refreshDone ? (
                  <span className="absolute right-0 top-full z-50 mt-2 whitespace-nowrap rounded-full border border-emerald-200/20 bg-[rgba(4,17,24,0.94)] px-3 py-1 text-xs text-emerald-100 shadow-[0_12px_32px_rgba(0,0,0,0.35)]">
                    已刷新
                  </span>
                ) : null}
              </div>
              <div className="flex shrink-0 items-center gap-2 rounded-full border border-cyan-300/16 bg-cyan-300/[0.06] px-3 py-2 text-xs text-white/75 shadow-[0_0_24px_rgba(85,215,255,0.06)]">
                <BookOpenText className="size-4 text-cyan-100/78" />
                <select
                  value={selectedKnowledgeBaseId ?? ""}
                  onChange={(event) => setSelectedKnowledgeBaseId(event.target.value || null)}
                  disabled={knowledgeBases.length === 0}
                  className="w-[min(12rem,52vw)] min-w-0 bg-transparent text-sm text-white outline-none lg:w-auto lg:min-w-[11rem]"
                >
                  {knowledgeBases.length === 0 ? (
                    <option value="" className="bg-[#081126] text-white">
                      暂无资料库
                    </option>
                  ) : null}
                  {knowledgeBases.map((knowledgeBase) => (
                    <option key={knowledgeBase.id} value={knowledgeBase.id} className="bg-[#081126] text-white">
                      {knowledgeBase.name}
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  aria-label="删除当前资料库"
                  title="删除当前资料库"
                  onClick={() => {
                    if (!selectedKnowledgeBaseId || !selectedKnowledgeBase) {
                      return;
                    }
                    setDeleteKnowledgeBaseResult(null);
                    deleteKnowledgeBaseMutation.reset();
                    setKnowledgeBaseToDelete({ id: selectedKnowledgeBaseId, name: selectedKnowledgeBase.name });
                  }}
                  disabled={!selectedKnowledgeBaseId || deleteKnowledgeBaseMutation.isPending}
                  className="grid size-8 place-items-center rounded-full border border-rose-200/20 text-rose-100/70 transition hover:border-rose-200/45 hover:bg-rose-300/10 hover:text-rose-50 disabled:pointer-events-none disabled:opacity-40"
                >
                  <Trash2 className="size-4" />
                </button>
              </div>
              <Button type="button" variant="outline" className="shrink-0 rounded-full" onClick={() => setCreateOpen(true)}>
                <FolderPlus data-icon="inline-start" />
                新建资料库
              </Button>
              <div className="kg-micro-chip !hidden shrink-0 rounded-full px-3 py-2 text-xs sm:!inline-flex">
                <Sparkles data-icon="inline-start" />
                智能检索问答
              </div>
              <div className="kg-micro-chip !hidden shrink-0 rounded-full px-3 py-2 text-xs sm:!inline-flex">
                <TerminalSquare data-icon="inline-start" />
                资料已连接
              </div>
            </div>
          </div>
        </header>

        <main className="px-4 pb-5 pt-[12.5rem] sm:pt-[10.5rem] lg:px-7 lg:pb-7 lg:pt-[8.5rem]">
          <div className="flex w-full flex-col gap-8">{children}</div>
        </main>
      </div>

      <Dialog open={createOpen} onOpenChange={setCreateOpen}>
        <DialogContent className="max-w-md border border-white/10 bg-[rgba(3,7,20,0.88)] p-0 text-white shadow-[0_30px_80px_rgba(0,0,0,0.4)] backdrop-blur-2xl">
          <DialogHeader className="border-b border-white/8 px-6 py-5">
            <DialogTitle>新建资料库空间</DialogTitle>
            <DialogDescription>创建资料库看板、图谱、搜索和问答上下文。资料文件会统一进入当前资料库存储文件夹。</DialogDescription>
          </DialogHeader>
          <form
            className="space-y-4 px-6 py-5"
            onSubmit={async (event) => {
              event.preventDefault();
              const name = nextKnowledgeBaseName.trim();
              if (!name) {
                return;
              }
              await createKnowledgeBaseSpace({ name });
              setNextKnowledgeBaseName("");
              setCreateOpen(false);
              if (window.localStorage.getItem(CREATE_KB_TUTORIAL_STORAGE_KEY) !== "true") {
                setHideCreateTutorialDraft(false);
                setCreateTutorialOpen(true);
              }
            }}
          >
            <label className="flex flex-col gap-2">
              <span className="text-xs uppercase tracking-[0.24em] text-cyan-100/46">资料库名称</span>
              <Input
                value={nextKnowledgeBaseName}
                onChange={(event) => setNextKnowledgeBaseName(event.target.value)}
                placeholder="线性代数"
                className="h-12 rounded-2xl border-white/10 bg-white/[0.04] px-4 text-white placeholder:text-white/28"
              />
            </label>
            <div className="flex items-center justify-end gap-2">
              <Button type="button" variant="outline" className="rounded-full" onClick={() => setCreateOpen(false)}>
                取消
              </Button>
              <Button type="submit" className="rounded-full" disabled={isCreating || !nextKnowledgeBaseName.trim()}>
                <FolderPlus data-icon="inline-start" />
                {isCreating ? "创建中" : "创建"}
              </Button>
            </div>
          </form>
        </DialogContent>
      </Dialog>

      <Dialog open={createTutorialOpen} onOpenChange={setCreateTutorialOpen}>
        <DialogContent className="max-h-[calc(100vh-2rem)] w-[min(44rem,calc(100vw-2rem))] overflow-hidden border border-white/10 bg-[rgba(3,7,20,0.94)] p-0 text-white shadow-[0_30px_80px_rgba(0,0,0,0.42)] backdrop-blur-2xl sm:!max-w-2xl">
          <DialogHeader className="border-b border-white/8 px-6 py-5 pr-14">
            <DialogTitle>新资料库操作流程</DialogTitle>
            <DialogDescription className="break-words text-cyan-100/70">
              新建资料库默认使用内置课程配置档；后续可以在设置页复制预设并切换为自定义配置档。
            </DialogDescription>
          </DialogHeader>
          <div className="max-h-[calc(100vh-10rem)] space-y-4 overflow-y-auto px-6 py-5">
            <div className="grid gap-3">
              <div className="flex min-w-0 gap-4 rounded-2xl border border-cyan-200/15 bg-cyan-300/[0.06] p-4">
                <div className="grid size-10 shrink-0 place-items-center rounded-full border border-cyan-200/20 bg-cyan-200/[0.08] text-cyan-100">
                  <TerminalSquare className="size-5" />
                </div>
                <div className="min-w-0">
                  <p className="text-sm font-semibold text-white">环境设置</p>
                  <p className="mt-2 break-words text-xs leading-5 text-white/60">确认模型桥、聊天模型、向量模型、并发，以及 Redis、PostgreSQL、Qdrant 运行状态。</p>
                </div>
              </div>
              <div className="flex min-w-0 gap-4 rounded-2xl border border-violet-200/15 bg-violet-300/[0.06] p-4">
                <div className="grid size-10 shrink-0 place-items-center rounded-full border border-violet-200/20 bg-violet-200/[0.08] text-violet-100">
                  <Settings className="size-5" />
                </div>
                <div className="min-w-0">
                  <p className="text-sm font-semibold text-white">配置档设置</p>
                  <p className="mt-2 break-words text-xs leading-5 text-white/60">默认配置档保持现有交互方式；自定义资料库先复制预设，再编辑界面标签、提示词和对话偏好。</p>
                </div>
              </div>
              <div className="flex min-w-0 gap-4 rounded-2xl border border-emerald-200/15 bg-emerald-300/[0.06] p-4">
                <div className="grid size-10 shrink-0 place-items-center rounded-full border border-emerald-200/20 bg-emerald-200/[0.08] text-emerald-100">
                  <Upload className="size-5" />
                </div>
                <div className="min-w-0">
                  <p className="text-sm font-semibold text-white">导入全流程</p>
                  <p className="mt-2 break-words text-xs leading-5 text-white/60">上传文件后执行解析、构图、检索验证和问答检查；配置档切换只影响之后的新任务。</p>
                </div>
              </div>
            </div>
            <label className="flex items-center gap-3 rounded-2xl border border-white/10 bg-white/[0.04] px-4 py-3 text-sm text-white/70">
              <input
                type="checkbox"
                checked={hideCreateTutorialDraft}
                onChange={(event) => setHideCreateTutorialDraft(event.target.checked)}
                className="size-4 rounded border-white/20 bg-black/30 accent-cyan-300"
              />
              不再显示这个教程
            </label>
            <div className="flex justify-end">
              <Button
                type="button"
                className="rounded-full"
                onClick={() => {
                  window.localStorage.setItem(CREATE_KB_TUTORIAL_STORAGE_KEY, hideCreateTutorialDraft ? "true" : "false");
                  setCreateTutorialOpen(false);
                }}
              >
                我已了解
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      <Dialog
        open={Boolean(knowledgeBaseToDelete)}
        onOpenChange={(open) => {
          if (!open && !deleteKnowledgeBaseMutation.isPending) {
            setKnowledgeBaseToDelete(null);
            setDeleteKnowledgeBaseResult(null);
          }
        }}
      >
        <DialogContent className="max-w-md border border-white/10 bg-[rgba(3,7,20,0.92)] p-0 text-white shadow-[0_30px_80px_rgba(0,0,0,0.4)] backdrop-blur-2xl" showCloseButton={!deleteKnowledgeBaseMutation.isPending}>
          <DialogHeader className="border-b border-white/8 px-6 py-5">
            <DialogTitle>删除资料库</DialogTitle>
            <DialogDescription>
              {knowledgeBaseToDelete ? `将从文件存储、PostgreSQL、Qdrant 和图谱表中删除“${knowledgeBaseToDelete.name}”。` : ""}
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 px-6 py-5">
            {deleteKnowledgeBaseMutation.isPending ? (
              <div>
                <p className="text-sm text-white/72">正在删除资料库数据...</p>
                <div className="mt-3 h-2 overflow-hidden rounded-full bg-white/8">
                  <div className="h-full w-2/3 animate-pulse rounded-full bg-[linear-gradient(90deg,#fb7185,#fbbf24,#fb7185)]" />
                </div>
              </div>
            ) : deleteKnowledgeBaseResult ? (
              <p className="rounded-2xl border border-emerald-200/16 bg-emerald-300/[0.055] px-4 py-3 text-sm leading-6 text-emerald-50/78">{deleteKnowledgeBaseResult}</p>
            ) : (
              <p className="rounded-2xl border border-rose-200/16 bg-rose-300/[0.055] px-4 py-3 text-sm leading-6 text-rose-50/78">
                这是不可逆操作，界面内无法撤销。
              </p>
            )}
            {deleteKnowledgeBaseMutation.error ? <p className="text-sm text-rose-100/78">{(deleteKnowledgeBaseMutation.error as Error).message}</p> : null}
            <div className="flex justify-end gap-2">
              <Button
                type="button"
                variant="outline"
                className="rounded-full"
                disabled={deleteKnowledgeBaseMutation.isPending}
                onClick={() => {
                  setKnowledgeBaseToDelete(null);
                  setDeleteKnowledgeBaseResult(null);
                }}
              >
                {deleteKnowledgeBaseResult ? "关闭" : "取消"}
              </Button>
              {!deleteKnowledgeBaseResult ? (
                <Button
                  type="button"
                  className="rounded-full"
                  disabled={!knowledgeBaseToDelete || deleteKnowledgeBaseMutation.isPending}
                  onClick={() => {
                    if (knowledgeBaseToDelete) {
                      deleteKnowledgeBaseMutation.mutate(knowledgeBaseToDelete.id);
                    }
                  }}
                >
                  {deleteKnowledgeBaseMutation.isPending ? "删除中..." : "删除"}
                </Button>
              ) : null}
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
