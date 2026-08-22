"use client";

import type { AgentTraceEventPayload } from "@course-kg/shared";
import { useQuery } from "@tanstack/react-query";

import { EvidenceEvaluatorAudit, GrayZoneAuditDetails, GrayZoneAuditFailure, QueryFacetPosteriorAudit } from "@/components/gray-zone-audit";
import { LoadingBlock } from "@/components/query-state";
import { fetchRetrievalTraceSteps } from "@/lib/api";

type ApiError = Error & {
  status?: number;
  structured?: { code?: string; title?: string; message?: string };
};

function retrievalAuditError(error: Error): { title: string; message: string } {
  const apiError = error as ApiError;
  const status = apiError.status;
  const code = apiError.structured?.code;
  return {
    title: status === 409 ? "Gray-zone 持久化轨迹校验冲突" : "Gray-zone 完整轨迹读取失败",
    message: [status ? `HTTP ${status}` : "", code || "", apiError.structured?.message || error.message].filter(Boolean).join(" · "),
  };
}

export function QARetrievalAudit({
  traceId,
  agentTrace,
  retrievalExpected = true,
}: {
  traceId?: string | null;
  agentTrace: AgentTraceEventPayload[];
  retrievalExpected?: boolean;
}) {
  const traceQuery = useQuery({
    queryKey: ["qa-retrieval-trace-steps", traceId],
    queryFn: () => fetchRetrievalTraceSteps(traceId as string),
    enabled: Boolean(traceId),
    retry: false,
  });

  return (
    <div data-testid="qa-retrieval-audit" className="mt-4 grid gap-3">
      <EvidenceEvaluatorAudit trace={agentTrace} />
      {!traceId ? (
        retrievalExpected ? (
          <GrayZoneAuditFailure
            title="Gray-zone 完整轨迹缺失"
            message="该 Agent 回答没有提供 retrieval_trace_id，无法读取持久化图检索轨迹并证明灰区模型调用数为零。"
          />
        ) : (
          <p className="rounded-2xl border border-white/8 bg-white/[0.025] p-4 text-xs text-white/42">本次非检索路线未生成 retrieval trace。</p>
        )
      ) : traceQuery.isLoading ? (
        <LoadingBlock rows={2} />
      ) : traceQuery.error instanceof Error ? (
        <GrayZoneAuditFailure {...retrievalAuditError(traceQuery.error)} />
      ) : traceQuery.data ? (
        <>
          <QueryFacetPosteriorAudit calibration={traceQuery.data.trace_diagnostics.query_facet_posterior_calibration} />
          <GrayZoneAuditDetails trace={traceQuery.data} />
        </>
      ) : (
        <GrayZoneAuditFailure message="retrieval trace 请求结束但没有返回 payload。" />
      )}
    </div>
  );
}
