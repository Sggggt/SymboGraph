"use client";

import type { Citation } from "@course-kg/shared";
import { FileText } from "lucide-react";

import { MarkdownRenderer } from "@/components/markdown-renderer";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface CitationCardProps {
  citation: Citation;
  index: number;
}

export function CitationCard({ citation, index }: CitationCardProps) {
  return (
    <Card className="border-white/10 bg-white/[0.03] text-white">
      <CardHeader>
        <CardTitle className="flex items-start justify-between gap-3">
          <span className="min-w-0 truncate">{citation.document_title}</span>
          <Badge variant="outline">#{index + 1}</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <MarkdownRenderer content={citation.snippet} className="text-white/68" />
        <div className="flex flex-wrap gap-2 text-xs text-white/48">
          <Badge variant="secondary">{citation.partition ?? "通用"}</Badge>
          {citation.section ? <Badge variant="outline">{citation.section}</Badge> : null}
          {citation.page_number ? <Badge variant="outline">第 {citation.page_number} 页</Badge> : null}
        </div>
        <p className="flex items-center gap-2 truncate text-xs text-white/35">
          <FileText data-icon="inline-start" />
          {citation.source_path}
        </p>
      </CardContent>
    </Card>
  );
}
