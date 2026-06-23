"use client";

import type { AgentTraceEventPayload } from "@course-kg/shared";

import { AgentTraceStream } from "@/components/agent-trace-stream";

interface AgentTracePanelProps {
  trace: AgentTraceEventPayload[];
}

export function AgentTracePanel({ trace }: AgentTracePanelProps) {
  return (
    <div className="min-h-0 text-white">
      <div className="mb-4 flex items-center justify-between gap-3">
        <h3 className="text-sm font-semibold text-white/82">智能体事件流</h3>
        <span className="font-mono text-xs text-white/42">{trace.length} events</span>
      </div>
      <AgentTraceStream trace={trace} defaultExpanded compact />
    </div>
  );
}
