"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { AgentResponse, AgentTraceEventPayload, Citation, SessionSummary } from "@course-kg/shared";
import { motion } from "framer-motion";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Archive,
  BrainCircuit,
  ChevronRight,
  CircleDot,
  FileText,
  History,
  Layers3,
  Loader2,
  Plus,
  Send,
  ShieldCheck,
  Sparkles,
  Trash2,
} from "lucide-react";

import { CitationCard } from "@/components/citation-card";
import { useCourseContext } from "@/components/course-context";
import { MarkdownRenderer } from "@/components/markdown-renderer";
import { ErrorBlock, LoadingBlock } from "@/components/query-state";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { Textarea } from "@/components/ui/textarea";
import { deleteSession, fetchDashboard, fetchSessionMessages, fetchSessions, streamAnswer } from "@/lib/api";
import { evidenceFirstTraceFallbackSteps, traceAuditSummary, traceNodeLabel } from "@/lib/agent-trace";
import { cn } from "@/lib/utils";
import { useLocalStorage } from "@/hooks/use-local-storage";

type ChatTurn = {
  role: "user" | "assistant";
  content: string;
  run_id?: string | null;
  route?: string | null;
  citations?: Citation[];
  trace?: AgentTraceEventPayload[];
};

const fallbackSuggestions = [
  "总结这批资料最核心的知识结构",
  "结合本地资料解释一个重要概念",
  "找出资料库中容易混淆的概念并比较",
  "基于资料引用给我一份阅读路线",
];

function answerModelLabel(latestRun: AgentResponse | null): string {
  const audit = latestRun?.answer_model_audit;
  if (!audit) {
    return "模型：等待回答";
  }
  if (audit.external_called) {
    return `模型：${audit.model ?? audit.provider}`;
  }
  if (audit.skipped_reason === "clarify_route") {
    return "模型：澄清分支未调用";
  }
  if (audit.skipped_reason === "direct_answer_route") {
    return "模型：直接回答分支未调用";
  }
  return "模型：未调用";
}

function buildCourseSuggestions(tree: Array<{ title: string; children?: Array<{ title: string }> }> | undefined): string[] {
  const chapters = tree?.map((node) => node.title).filter(Boolean) ?? [];
  const documents = tree?.flatMap((node) => node.children?.map((child) => child.title) ?? []).filter(Boolean) ?? [];
  const suggestions = [
    chapters[0] ? `总结 ${chapters[0]} 的核心内容` : "",
    chapters[1] ? `比较 ${chapters[0]} 和 ${chapters[1]} 的联系` : "",
    documents[0] ? `根据 ${documents[0]} 生成整理提纲` : "",
    chapters[0] ? `从本地资料中找出 ${chapters[0]} 的关键概念` : "",
  ].filter(Boolean);
  return suggestions.length ? suggestions.slice(0, 4) : fallbackSuggestions;
}

function normalizeMessages(messages: Array<Record<string, unknown>>): ChatTurn[] {
  return messages
    .filter((item) => item.role === "user" || item.role === "assistant")
    .map((item) => ({
      role: item.role as "user" | "assistant",
      content: String(item.content ?? ""),
      run_id: typeof item.run_id === "string" ? item.run_id : null,
      citations: Array.isArray(item.citations) ? (item.citations as Citation[]) : undefined,
    }));
}

function ChatHeader({
  grounded,
  chapterList,
  latestRun,
}: {
  grounded: boolean;
  chapterList: string;
  latestRun: AgentResponse | null;
}) {
  return (
    <div className="mx-auto flex w-full max-w-6xl flex-wrap items-center justify-between gap-3 px-1">
      <div className="min-w-0">
        <p className="section-kicker">资料库智能问答</p>
        <h2 className="mt-1 text-2xl font-semibold text-white lg:text-3xl">向推理链路提问</h2>
      </div>
      <div className="flex w-full flex-wrap gap-2">
        <span className="kg-micro-chip rounded-full px-3 py-2 text-xs">
          <ShieldCheck data-icon="inline-start" />
          {grounded ? "已接入证据" : "降级模式"}
        </span>
        <span className="kg-micro-chip max-w-full truncate rounded-full px-3 py-2 text-xs">目录：{chapterList || "等待中"}</span>
        {latestRun?.route ? (
          <span className="kg-micro-chip rounded-full px-3 py-2 text-xs">
            <BrainCircuit data-icon="inline-start" />
            {latestRun.route}
          </span>
        ) : null}
        <span className="kg-micro-chip rounded-full px-3 py-2 text-xs">
          <BrainCircuit data-icon="inline-start" />
          {answerModelLabel(latestRun)}
        </span>
      </div>
    </div>
  );
}

function ChatActionRail({
  onOpenSessions,
  onOpenCitations,
  citationsCount,
}: {
  onOpenSessions: () => void;
  onOpenCitations: () => void;
  citationsCount: number;
}) {
  const actions = [
    { label: "会话", icon: History, onClick: onOpenSessions },
    { label: `引用 ${citationsCount}`, icon: FileText, onClick: onOpenCitations },
  ];

  return (
    <div className="fixed bottom-[11.5rem] right-4 z-40 flex flex-col gap-2 lg:bottom-auto lg:right-7 lg:top-[10rem]">
      {actions.map(({ label, icon: Icon, onClick }) => (
        <button
          key={label}
          type="button"
          onClick={onClick}
          className="group flex h-11 items-center justify-end gap-2 rounded-full border border-cyan-200/14 bg-[rgba(7,13,31,0.88)] px-3 text-xs text-white/68 shadow-[0_16px_44px_rgba(0,0,0,0.28),0_0_28px_rgba(86,217,255,0.06)] backdrop-blur-2xl transition hover:border-cyan-200/32 hover:bg-cyan-300/[0.08] hover:text-white"
        >
          <span className="hidden whitespace-nowrap sm:inline">{label}</span>
          <Icon className="size-4 text-cyan-100/72 transition group-hover:text-cyan-100" />
        </button>
      ))}
    </div>
  );
}

function SuggestionChips({ suggestions, onPick }: { suggestions: string[]; onPick: (value: string) => void }) {
  return (
    <div className="mx-auto flex max-w-3xl flex-wrap justify-center gap-2">
      {suggestions.map((suggestion) => (
        <motion.button
          key={suggestion}
          type="button"
          whileHover={{ y: -2 }}
          whileTap={{ scale: 0.98 }}
          onClick={() => onPick(suggestion)}
          className="kg-micro-chip rounded-full px-4 py-2 text-sm transition hover:border-cyan-200/30 hover:text-white"
        >
          {suggestion}
        </motion.button>
      ))}
    </div>
  );
}

function EmptyChatState({ suggestions, onPick }: { suggestions: string[]; onPick: (value: string) => void }) {
  return (
    <div className="grid min-h-[calc(100dvh-21rem)] place-items-center px-4 pb-44 pt-12 text-center">
      <div className="-translate-y-24">
        <div className="mx-auto grid size-16 place-items-center rounded-3xl border border-cyan-200/14 bg-cyan-300/[0.045] text-cyan-100 shadow-[0_0_42px_rgba(86,217,255,0.08)]">
          <Sparkles />
        </div>
        <h3 className="glow-text mt-6 text-3xl font-semibold text-white">开始一轮有证据支撑的资料问答</h3>
        <p className="mx-auto mt-3 max-w-2xl text-sm leading-7 text-white/56">
          系统会先召回基础证据，再选择锚点、规划证据链、受控图增强，并校验引用来源。
        </p>
        <div className="mt-7">
          <SuggestionChips suggestions={suggestions} onPick={onPick} />
        </div>
      </div>
    </div>
  );
}

function TraceSummaryStrip({ trace, isRunning }: { trace: AgentTraceEventPayload[]; isRunning: boolean }) {
  if (trace.length === 0 && !isRunning) {
    return null;
  }
  const latest = trace.at(-1);
  return (
    <div className="flex flex-wrap items-center gap-2 text-[11px] text-white/45">
      <span className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.025] px-2.5 py-1">
        {isRunning ? <span className="tech-dot" /> : <ShieldCheck className="size-3.5 text-cyan-100/60" />}
        {trace.length ? `${trace.length} 步轨迹` : "准备轨迹"}
      </span>
      {latest ? (
        <span className="min-w-0 truncate rounded-full border border-white/10 bg-white/[0.025] px-2.5 py-1">
          {traceNodeLabel(latest.node)}
        </span>
      ) : null}
    </div>
  );
}

function InlineTraceDisclosure({
  trace,
  isRunning = false,
}: {
  trace: AgentTraceEventPayload[];
  isRunning?: boolean;
}) {
  const [expanded, setExpanded] = useState(false);
  const steps: AgentTraceEventPayload[] = trace.length
    ? trace
    : evidenceFirstTraceFallbackSteps.map((node) => ({ node, status: "pending", document_ids: [], scores: {}, duration_ms: 0 }));

  if (trace.length === 0 && !isRunning) {
    return null;
  }

  return (
    <div className="mb-4 rounded-2xl border border-white/8 bg-white/[0.018] px-3 py-2">
      <button type="button" onClick={() => setExpanded((current) => !current)} className="flex w-full items-center justify-between gap-3 text-left">
        <TraceSummaryStrip trace={trace} isRunning={isRunning} />
        <span className="shrink-0 text-[11px] text-cyan-100/55">{expanded ? "收起" : "查看轨迹"}</span>
      </button>
      {expanded ? (
        <div className="mt-3 flex flex-col gap-2 border-t border-white/8 pt-3">
          {steps.map((event, index) => {
            const latest = isRunning && index === steps.length - 1;
            const auditSummary = traceAuditSummary(event.scores);
            return (
              <div key={event.id ?? `${event.node}-${index}`} className="rounded-xl border border-white/8 bg-black/10 px-3 py-2">
                <div className="flex items-center justify-between gap-3">
                  <span className={cn("min-w-0 truncate text-xs font-medium", latest ? "text-cyan-100" : "text-white/70")}>
                    {traceNodeLabel(event.node)}
                  </span>
                  <span className="shrink-0 font-mono text-[10px] text-white/34">{event.duration_ms}ms</span>
                </div>
                {event.output_summary ? <MarkdownRenderer content={event.output_summary} compact className="mt-1 text-xs leading-5 text-white/46" /> : null}
                {auditSummary.length ? <p className="mt-1 truncate text-[10px] text-white/34">{auditSummary.join(" / ")}</p> : null}
              </div>
            );
          })}
        </div>
      ) : null}
    </div>
  );
}

function MessageBubble({ turn, index, onOpenCitations }: { turn: ChatTurn; index: number; onOpenCitations: () => void }) {
  const isUser = turn.role === "user";
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: Math.min(index * 0.025, 0.18) }}
      className={cn("flex", isUser ? "justify-end" : "justify-start")}
    >
      <div
        className={cn(
          "relative max-w-[min(860px,92%)] px-1 py-3",
          isUser
            ? "rounded-[1.25rem] border border-cyan-200/12 bg-cyan-300/[0.045] px-5 shadow-[0_0_24px_rgba(86,217,255,0.035)]"
            : "w-full border-l border-cyan-200/18 pl-5 text-white",
        )}
      >
        <div className="mb-3 flex items-center gap-2 text-xs uppercase tracking-[0.2em] text-white/38">
          {isUser ? <CircleDot /> : <BrainCircuit />}
          {isUser ? "你" : turn.route ? `智能体 / ${turn.route}` : "智能体"}
        </div>
        {!isUser && turn.trace?.length ? <InlineTraceDisclosure trace={turn.trace} /> : null}
        <MarkdownRenderer content={turn.content} className={cn(isUser ? "text-white/78" : "text-white/74")} />
        {!isUser && turn.citations?.length ? (
          <button type="button" onClick={onOpenCitations} className="kg-micro-chip mt-4 rounded-full px-3 py-2 text-xs transition hover:border-cyan-200/30 hover:text-white">
            <FileText />
            {turn.citations.length} 条已验证来源
          </button>
        ) : null}
      </div>
    </motion.div>
  );
}

function GeneratingBubble({ content, trace }: { content: string; trace: AgentTraceEventPayload[] }) {
  return (
    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="flex justify-start">
      <div className="w-full max-w-[min(860px,92%)] border-l border-cyan-200/18 px-5 py-4 text-white">
        <div className="mb-3 flex items-center gap-2 text-xs uppercase tracking-[0.2em] text-cyan-100/50">
          <span className="tech-dot" />
          {content ? "正在输出" : "智能体运行中"}
        </div>
        {!content ? <InlineTraceDisclosure trace={trace} isRunning /> : null}
        {content ? (
          <div className="relative">
            <MarkdownRenderer content={content} className="pr-3 text-white/76" />
            <span className="stream-cursor">|</span>
          </div>
        ) : (
          <div className="flex items-center gap-2 text-sm text-white/56">
            <span className="signal-bars">
              <span />
              <span />
              <span />
              <span />
            </span>
            正在路由、检索并检查证据...
          </div>
        )}
      </div>
    </motion.div>
  );
}

function MessageList({
  turns,
  isGenerating,
  draftAnswer,
  trace,
  onPickSuggestion,
  onOpenCitations,
  suggestions,
}: {
  turns: ChatTurn[];
  isGenerating: boolean;
  draftAnswer: string;
  trace: AgentTraceEventPayload[];
  onPickSuggestion: (value: string) => void;
  onOpenCitations: () => void;
  suggestions: string[];
}) {
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const scrollFrameRef = useRef<number | null>(null);
  const previousTurnCountRef = useRef(turns.length);

  useEffect(() => {
    const hasNewTurn = turns.length !== previousTurnCountRef.current;
    previousTurnCountRef.current = turns.length;
    const distanceToBottom = document.documentElement.scrollHeight - window.innerHeight - window.scrollY;
    const shouldStickToBottom = distanceToBottom < 280;
    if (!hasNewTurn && (!isGenerating || !shouldStickToBottom)) {
      return undefined;
    }
    if (scrollFrameRef.current !== null) {
      window.cancelAnimationFrame(scrollFrameRef.current);
    }
    scrollFrameRef.current = window.requestAnimationFrame(() => {
      bottomRef.current?.scrollIntoView({ behavior: "auto", block: "end" });
      scrollFrameRef.current = null;
    });
    return () => {
      if (scrollFrameRef.current !== null) {
        window.cancelAnimationFrame(scrollFrameRef.current);
        scrollFrameRef.current = null;
      }
    };
  }, [turns.length, draftAnswer, isGenerating]);

  return (
    <div className="relative min-h-[calc(100dvh-21rem)]">
      {turns.length === 0 && !isGenerating ? (
        <EmptyChatState suggestions={suggestions} onPick={onPickSuggestion} />
      ) : (
        <div className="mx-auto flex max-w-5xl flex-col gap-8 px-1 pb-6 pt-4">
          {turns.map((turn, index) => (
            <MessageBubble key={`${turn.role}-${index}-${turn.run_id ?? "local"}`} turn={turn} index={index} onOpenCitations={onOpenCitations} />
          ))}
          {isGenerating ? <GeneratingBubble content={draftAnswer} trace={trace} /> : null}
          <div ref={bottomRef} className="h-52 shrink-0 md:h-56" />
        </div>
      )}
    </div>
  );
}

function ChatComposer({
  value,
  onChange,
  onSubmit,
  isPending,
  activeSessionId,
}: {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  isPending: boolean;
  activeSessionId: string | null;
}) {
  const handleSubmit = () => {
    if (isPending || !value.trim()) {
      return;
    }
    onSubmit();
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      className="pointer-events-none fixed inset-x-4 bottom-4 z-[45] lg:left-[calc(76px+1.75rem)] lg:right-7"
    >
      <div className="pointer-events-auto mx-auto w-full max-w-5xl">
        <div
          className={cn(
            "kg-scan-edge rounded-[1.7rem] border border-cyan-200/16 bg-[rgba(7,13,31,0.94)] p-2 shadow-[0_20px_70px_rgba(0,0,0,0.42),0_0_42px_rgba(86,217,255,0.08)] backdrop-blur-2xl",
            isPending && "border-cyan-100/24 shadow-[0_20px_70px_rgba(0,0,0,0.34),0_0_58px_rgba(86,217,255,0.14)]",
          )}
        >
          <div className="flex flex-col gap-3 rounded-[1.35rem] bg-[linear-gradient(135deg,rgba(86,217,255,0.065),rgba(122,95,255,0.035)_55%,rgba(0,0,0,0.12))] p-3">
            <div className="flex flex-wrap items-center gap-2 px-1">
              <span className="kg-micro-chip rounded-full px-2.5 py-1 text-[11px]">
                <Layers3 />
                资料库上下文
              </span>
              <span className="kg-micro-chip max-w-full truncate rounded-full px-2.5 py-1 text-[11px]">
                会话 {activeSessionId ? activeSessionId.slice(0, 8) : "新建"}
              </span>
            </div>
            <div className="flex items-end gap-3">
              <Textarea
                value={value}
                onChange={(event) => onChange(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    if (isPending) {
                      return;
                    }
                    handleSubmit();
                  }
                }}
                className="max-h-44 min-h-[72px] resize-none border-0 bg-transparent px-2 text-base text-white shadow-none placeholder:text-white/30 focus-visible:ring-0"
                placeholder="输入问题，系统会检索、评估、回答并给出引用..."
              />
              <Button type="button" size="icon-lg" className="rounded-full" onClick={handleSubmit} disabled={isPending || !value.trim()}>
                {isPending ? <Loader2 className="animate-spin" /> : <Send />}
                <span className="sr-only">提问</span>
              </Button>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}

function SessionsDrawer({
  open,
  onOpenChange,
  sessions,
  activeSessionId,
  onSelect,
  onDelete,
  onNew,
  isPending,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  sessions: SessionSummary[];
  activeSessionId: string | null;
  onSelect: (sessionId: string) => void | Promise<void>;
  onDelete: (sessionId: string) => void | Promise<void>;
  onNew: () => void;
  isPending: boolean;
}) {
  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="left" className="w-full border-white/10 bg-[rgba(3,7,20,0.78)] p-0 text-white backdrop-blur-2xl sm:max-w-md">
        <SheetHeader className="border-b border-white/8 p-6">
          <SheetTitle>会话</SheetTitle>
          <SheetDescription>资料库智能体的对话记忆。</SheetDescription>
        </SheetHeader>
        <div className="p-5">
          <Button
            type="button"
            className="w-full rounded-full"
            disabled={isPending}
            onClick={() => {
              if (isPending) {
                return;
              }
              onNew();
              onOpenChange(false);
            }}
          >
            <Plus data-icon="inline-start" />
            新建会话
          </Button>
        </div>
        <ScrollArea className="h-[calc(100dvh-10rem)] px-5 pb-5">
          <div className="flex flex-col gap-2">
            {sessions.map((session) => (
              <div
                key={session.id}
                className={cn(
                  "flex items-start gap-2 rounded-2xl border px-3 py-3 transition",
                  session.id === activeSessionId ? "border-cyan-200/28 bg-cyan-300/[0.075]" : "border-white/7 bg-white/[0.025] hover:border-cyan-200/22",
                  isPending && "pointer-events-none opacity-50",
                )}
              >
                <button
                  type="button"
                  disabled={isPending}
                  onClick={() => {
                    if (isPending) {
                      return;
                    }
                    onSelect(session.id);
                    onOpenChange(false);
                  }}
                  className="min-w-0 flex-1 text-left"
                >
                  <div className="flex items-center justify-between gap-3">
                    <span className="min-w-0 truncate text-sm font-medium text-white">{session.title ?? "未命名会话"}</span>
                    <ChevronRight className="text-white/35" />
                  </div>
                  {session.last_question ? <p className="mt-2 line-clamp-2 text-xs leading-5 text-white/45">{session.last_question}</p> : null}
                </button>
                <button
                  type="button"
                  aria-label="删除会话"
                  disabled={isPending}
                  onClick={() => onDelete(session.id)}
                  className="grid size-8 shrink-0 place-items-center rounded-full border border-white/8 text-white/45 transition hover:border-rose-200/30 hover:bg-rose-300/[0.08] hover:text-rose-100 disabled:cursor-not-allowed disabled:opacity-45"
                >
                  <Trash2 className="size-4" />
                </button>
              </div>
            ))}
          </div>
        </ScrollArea>
      </SheetContent>
    </Sheet>
  );
}

function CitationsDrawer({
  open,
  onOpenChange,
  citations,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  citations: Citation[];
}) {
  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="w-full border-white/10 bg-[rgba(3,7,20,0.78)] p-0 text-white backdrop-blur-2xl sm:max-w-xl">
        <SheetHeader className="border-b border-white/8 p-6">
          <SheetTitle>引用</SheetTitle>
          <SheetDescription>智能体回答使用的已验证证据卡片。</SheetDescription>
        </SheetHeader>
        <ScrollArea className="h-[calc(100dvh-8rem)] p-6">
          <div className="flex flex-col gap-3">
            {citations.length === 0 ? (
              <div className="kg-glass-line rounded-3xl px-6 py-10 text-center text-sm text-white/55">
                <Archive className="mx-auto mb-4 text-cyan-100/70" />
                有证据回答完成后会显示引用。
              </div>
            ) : (
              citations.map((citation, index) => <CitationCard key={`${citation.chunk_id}-${index}`} citation={citation} index={index} />)
            )}
          </div>
        </ScrollArea>
      </SheetContent>
    </Sheet>
  );
}

function QAWorkspaceContent({ selectedCourseId }: { selectedCourseId: string | null }) {
  const queryClient = useQueryClient();
  const storageScope = selectedCourseId ?? "unassigned";
  const dashboardQuery = useQuery({
    queryKey: ["dashboard", selectedCourseId],
    queryFn: () => fetchDashboard(selectedCourseId),
    enabled: Boolean(selectedCourseId),
  });
  const sessionsQuery = useQuery({
    queryKey: ["sessions", selectedCourseId],
    queryFn: () => fetchSessions(selectedCourseId),
    enabled: Boolean(selectedCourseId),
  });
  const [question, setQuestion] = useLocalStorage(`qa.question.${storageScope}`, "");
  const [activeSessionId, setActiveSessionId] = useLocalStorage<string | null>(`qa.sessionId.${storageScope}`, null);
  const [turns, setTurns] = useLocalStorage<ChatTurn[]>(`qa.turns.${storageScope}`, []);
  const [draftAnswer, setDraftAnswer] = useLocalStorage(`qa.draftAnswer.${storageScope}`, "");
  const [citations, setCitations] = useLocalStorage<Citation[]>(`qa.citations.${storageScope}`, []);
  const [trace, setTrace] = useLocalStorage<AgentTraceEventPayload[]>(`qa.trace.${storageScope}`, []);
  const [latestRun, setLatestRun] = useLocalStorage<AgentResponse | null>(`qa.latestRun.${storageScope}`, null);
  const [streamError, setStreamError] = useState<string | null>(null);
  const [sessionsOpen, setSessionsOpen] = useState(false);
  const [citationsOpen, setCitationsOpen] = useState(false);

  const askMutation = useMutation({
    mutationFn: async () => {
      const nextQuestion = question.trim();
      if (!nextQuestion) {
        return;
      }
      setStreamError(null);
      setDraftAnswer("");
      setCitations([]);
      setTrace([]);
      setLatestRun(null);
      const nextTraceEvents: AgentTraceEventPayload[] = [];
      setTurns((current) => [...current, { role: "user", content: nextQuestion }]);
      setQuestion("");
      try {
        await streamAnswer(
          { question: nextQuestion, session_id: activeSessionId, course_id: selectedCourseId, top_k: 6 },
          {
            onTrace: (event) => {
              nextTraceEvents.push(event);
              setTrace((current) => [...current, event]);
            },
            onToken: (token) => setDraftAnswer((current) => `${current}${token}`),
            onCitations: (next) => setCitations(next),
            onFinal: (response) => {
              const finalTrace = response.trace.length ? response.trace : nextTraceEvents;
              setLatestRun(response);
              setActiveSessionId(response.session_id);
              setCitations(response.citations);
              setDraftAnswer("");
              setTrace(finalTrace);
              setTurns((current) => [
                ...current,
                {
                  role: "assistant",
                  content: response.answer,
                  run_id: response.run_id,
                  route: response.route,
                  citations: response.citations,
                  trace: finalTrace,
                },
              ]);
              void queryClient.invalidateQueries({ queryKey: ["sessions", selectedCourseId] });
              void queryClient.invalidateQueries({ queryKey: ["session-messages", response.session_id] });
            },
            onError: (message) => setStreamError(message),
          },
        );
      } catch (error) {
        setStreamError(error instanceof Error ? error.message : String(error));
      }
    },
  });

  const deleteSessionMutation = useMutation({
    mutationFn: (sessionId: string) => deleteSession(sessionId),
    onSuccess: async (_data, sessionId) => {
      if (sessionId === activeSessionId) {
        setActiveSessionId(null);
        setTurns([]);
        setDraftAnswer("");
        setCitations([]);
        setTrace([]);
        setLatestRun(null);
        setQuestion("");
      }
      await queryClient.invalidateQueries({ queryKey: ["sessions", selectedCourseId] });
    },
  });

  const chapterList = useMemo(() => dashboardQuery.data?.tree.map((node) => node.title).join(" / ") ?? "", [dashboardQuery.data]);
  const suggestions = useMemo(() => buildCourseSuggestions(dashboardQuery.data?.tree), [dashboardQuery.data?.tree]);

  if (dashboardQuery.isLoading) {
    return <LoadingBlock rows={4} />;
  }
  if (dashboardQuery.error) {
    return <ErrorBlock message={(dashboardQuery.error as Error).message} />;
  }

  return (
    <div className="kg-page relative -mx-4 -my-5 min-h-[calc(100dvh-4.25rem)] px-4 pb-52 pt-5 lg:-mx-7 lg:-my-7 lg:px-7 lg:pt-7">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_50%_0%,rgba(86,217,255,0.11),transparent_34%),radial-gradient(circle_at_88%_20%,rgba(124,92,255,0.11),transparent_30%),linear-gradient(rgba(120,180,255,0.026)_1px,transparent_1px),linear-gradient(90deg,rgba(120,180,255,0.023)_1px,transparent_1px)] bg-[size:auto,auto,48px_48px,48px_48px]" />
      <div className="pointer-events-none absolute inset-x-0 bottom-0 h-64 bg-gradient-to-t from-[#030714] via-[#030714]/88 to-transparent" />
      <div className="relative z-10 flex flex-col gap-7">
        <ChatHeader
          grounded={!dashboardQuery.data?.degraded_mode}
          chapterList={chapterList}
          latestRun={latestRun}
        />
        <ChatActionRail
          onOpenSessions={() => setSessionsOpen(true)}
          onOpenCitations={() => setCitationsOpen(true)}
          citationsCount={citations.length}
        />

        <main className="mx-auto w-full max-w-6xl">
          {streamError ? <ErrorBlock message={streamError} /> : null}
          <MessageList
            turns={turns}
            isGenerating={askMutation.isPending}
            draftAnswer={draftAnswer}
            trace={trace}
            onPickSuggestion={setQuestion}
            onOpenCitations={() => setCitationsOpen(true)}
            suggestions={suggestions}
          />
        </main>

        <ChatComposer
          value={question}
          onChange={setQuestion}
          onSubmit={() => {
            if (askMutation.isPending) {
              return;
            }
            askMutation.mutate();
          }}
          isPending={askMutation.isPending}
          activeSessionId={activeSessionId}
        />
      </div>

      <SessionsDrawer
        open={sessionsOpen}
        onOpenChange={setSessionsOpen}
        sessions={sessionsQuery.data ?? []}
        activeSessionId={activeSessionId}
        onDelete={(sessionId) => deleteSessionMutation.mutate(sessionId)}
        onSelect={async (sessionId) => {
          setActiveSessionId(sessionId);
          setDraftAnswer("");
          setCitations([]);
          setTrace([]);
          setLatestRun(null);
          const response = await fetchSessionMessages(sessionId);
          const nextTurns = normalizeMessages(response.messages);
          setTurns(nextTurns);
          const latestAssistant = [...nextTurns].reverse().find((turn) => turn.role === "assistant");
          setCitations(latestAssistant?.citations ?? []);
        }}
        onNew={() => {
          setActiveSessionId(null);
          setTurns([]);
          setDraftAnswer("");
          setCitations([]);
          setTrace([]);
          setLatestRun(null);
          setQuestion("");
        }}
        isPending={askMutation.isPending}
      />
      <CitationsDrawer open={citationsOpen} onOpenChange={setCitationsOpen} citations={citations} />
    </div>
  );
}

export function QAWorkspace() {
  const { selectedCourseId } = useCourseContext();
  return <QAWorkspaceContent key={selectedCourseId ?? "unassigned"} selectedCourseId={selectedCourseId} />;
}
