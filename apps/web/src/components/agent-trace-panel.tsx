"use client";

import type { AgentTraceEventPayload } from "@course-kg/shared";
import { Activity, CheckCircle2, Clock3, XCircle } from "lucide-react";

import { MarkdownRenderer } from "@/components/markdown-renderer";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { groupTraceEvents, traceAuditSummary, traceNodeLabel } from "@/lib/agent-trace";

interface AgentTracePanelProps {
  trace: AgentTraceEventPayload[];
}

function traceBadgeVariant(event: AgentTraceEventPayload): "default" | "secondary" | "destructive" | "outline" {
  if (event.status === "failed" || event.node === "error") return "destructive";
  if (event.node === "repair_executed") return "outline";
  if (event.node === "citation_verification") return "default";
  return "secondary";
}

export function AgentTracePanel({ trace }: AgentTracePanelProps) {
  const groups = groupTraceEvents(trace);
  return (
    <Card className="min-h-0 border-white/10 bg-white/[0.03] text-white">
      <CardHeader>
        <CardTitle className="flex items-center justify-between gap-3">
          <span>智能体轨迹</span>
          <Badge variant="outline">{trace.length} 步</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="min-h-0">
        <ScrollArea className="h-[360px] pr-3">
          <div className="flex flex-col gap-4">
            {trace.length === 0 ? (
              <div className="rounded-lg border border-white/10 px-4 py-5 text-sm text-white/55">
                智能体开始运行后，这里会显示规划、遍历、证据包、引用验证和修复轨迹。
              </div>
            ) : (
              groups.map((group) => (
                <section key={group.key} className="space-y-2">
                  <div className="flex items-center justify-between gap-2">
                    <p className="text-xs font-semibold text-white/72">{group.label}</p>
                    <Badge variant="outline" className="text-[10px] text-white/55">
                      {group.events.length} 项
                    </Badge>
                  </div>
                  {group.events.map((event, index) => {
                    const Icon = event.status === "failed" ? XCircle : event.duration_ms > 0 ? CheckCircle2 : Activity;
                    const auditSummary = traceAuditSummary(event.scores);
                    return (
                      <div key={event.id ?? `${event.node}-${index}`} className="rounded-lg border border-white/10 bg-black/10 p-4">
                        <div className="flex items-start justify-between gap-3">
                          <div className="flex min-w-0 items-center gap-2">
                            <Icon data-icon="inline-start" />
                            <span className="truncate text-sm font-medium">{traceNodeLabel(event.node)}</span>
                            {event.status === "failed" ? (
                              <Badge variant="destructive" className="text-[10px]">
                                失败
                              </Badge>
                            ) : null}
                          </div>
                          <Badge variant={traceBadgeVariant(event)}>
                            <Clock3 data-icon="inline-start" />
                            {event.duration_ms} ms
                          </Badge>
                        </div>
                        {event.output_summary ? <MarkdownRenderer content={event.output_summary} compact className="mt-3 text-white/62" /> : null}
                        {auditSummary.length ? (
                          <div className="mt-3 flex flex-wrap gap-2">
                            {auditSummary.map((item) => (
                              <Badge key={item} variant="outline" className="max-w-full truncate text-[10px] text-white/58">
                                {item}
                              </Badge>
                            ))}
                          </div>
                        ) : null}
                        {event.document_ids.length > 0 ? <p className="mt-2 text-xs text-white/38">触达 {event.document_ids.length} 个片段</p> : null}
                      </div>
                    );
                  })}
                </section>
              ))
            )}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
}
