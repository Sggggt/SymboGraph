"use client";

import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { BookMarked, Radar } from "lucide-react";

import { useCourseContext } from "@/components/course-context";
import { fetchConcepts } from "@/lib/api";
import { ErrorBlock, LoadingBlock } from "@/components/query-state";
import { useLocalStorage } from "@/hooks/use-local-storage";

export function ConceptsWorkspace() {
  const { selectedCourseId } = useCourseContext();
  const storageScope = selectedCourseId ?? "unassigned";
  const { data, isLoading, error } = useQuery({
    queryKey: ["concepts", selectedCourseId],
    queryFn: () => fetchConcepts(selectedCourseId),
    enabled: Boolean(selectedCourseId),
  });
  const [keyword, setKeyword] = useLocalStorage(`concepts.keyword.${storageScope}`, "");
  const filtered = useMemo(
    () => (data ?? []).filter((item) => item.name.toLowerCase().includes(keyword.toLowerCase())),
    [data, keyword],
  );
  const [selectedId, setSelectedId] = useLocalStorage<string | null>(`concepts.selectedId.${storageScope}`, null);
  const selected = filtered.find((item) => item.concept_id === selectedId) ?? filtered[0] ?? null;

  if (isLoading) {
    return <LoadingBlock rows={4} />;
  }
  if (error) {
    return <ErrorBlock message={(error as Error).message} />;
  }

  return (
    <div className="kg-page grid gap-6 xl:grid-cols-[0.8fr_1.2fr]">
      <section className="glass-panel kg-scroll-panel rounded-[30px] p-6">
        <div className="flex items-center justify-between">
          <div>
            <p className="section-kicker">概念浏览</p>
            <h2 className="mt-2 text-3xl font-semibold text-white">知识点目录</h2>
          </div>
          <BookMarked className="size-5 text-cyan-200" />
        </div>
        <div className="mt-5">
          <input
            value={keyword}
            onChange={(event) => setKeyword(event.target.value)}
            className="h-12 w-full rounded-full border border-white/10 bg-white/[0.05] px-5 text-white outline-none placeholder:text-white/28"
            placeholder="筛选概念..."
          />
        </div>
        <div className="mt-6 space-y-3">
          {filtered.map((concept) => {
            const active = selected?.concept_id === concept.concept_id;
            return (
              <button
                key={concept.concept_id}
                type="button"
                onClick={() => setSelectedId(concept.concept_id)}
                className={`block w-full rounded-[22px] border px-5 py-4 text-left transition ${
                  active
                    ? "border-cyan-300/35 bg-cyan-300/[0.08]"
                    : "border-white/8 bg-white/[0.03] hover:border-cyan-300/25 hover:bg-cyan-300/[0.05]"
                }`}
              >
                <div className="flex items-center justify-between gap-4">
                  <p className="text-base font-medium text-white">{concept.name}</p>
                  <span className="rounded-full border border-white/10 px-3 py-1 text-xs text-white/55">{concept.related_concepts.length}</span>
                </div>
                <div className="mt-3 flex flex-wrap gap-2">
                  {concept.chapter_refs.slice(0, 3).map((chapter) => (
                    <span key={chapter} className="rounded-full border border-white/10 px-3 py-1 text-xs text-cyan-100/72">
                      {chapter}
                    </span>
                  ))}
                </div>
              </button>
            );
          })}
        </div>
      </section>

      <motion.section initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="glass-panel kg-scroll-panel rounded-[30px] p-6">
        {selected ? (
          <>
            <div className="flex items-center justify-between">
              <div>
                <p className="section-kicker">概念详情</p>
                <h2 className="mt-2 text-4xl font-semibold text-white">{selected.name}</h2>
              </div>
              <Radar className="size-5 text-cyan-200" />
            </div>

            <div className="mt-6 grid gap-5 lg:grid-cols-[1.1fr_0.9fr]">
              <div className="space-y-5">
                <div className="rounded-[24px] border border-white/8 bg-white/[0.03] p-5">
                  <p className="text-xs uppercase tracking-[0.28em] text-white/45">摘要</p>
                  <p className="mt-4 text-sm leading-8 text-white/72">{selected.summary || "当前概念还没有足够长的摘要，建议补充更多本地原件后重新抽取。"}</p>
                </div>
                <div className="rounded-[24px] border border-white/8 bg-white/[0.03] p-5">
                  <p className="text-xs uppercase tracking-[0.28em] text-white/45">别名</p>
                  <div className="mt-4 flex flex-wrap gap-2">
                    {(selected.aliases.length > 0 ? selected.aliases : [selected.name]).map((alias) => (
                      <span key={alias} className="rounded-full border border-white/10 px-3 py-1 text-sm text-cyan-50/74">
                        {alias}
                      </span>
                    ))}
                  </div>
                </div>
              </div>

              <div className="space-y-5">
                <div className="rounded-[24px] border border-white/8 bg-white/[0.03] p-5">
                  <p className="text-xs uppercase tracking-[0.28em] text-white/45">目录引用</p>
                  <div className="mt-4 flex flex-wrap gap-2">
                    {selected.chapter_refs.map((chapter) => (
                      <span key={chapter} className="rounded-full border border-white/10 px-3 py-1 text-sm text-cyan-50/74">
                        {chapter}
                      </span>
                    ))}
                  </div>
                </div>
                <div className="rounded-[24px] border border-white/8 bg-white/[0.03] p-5">
                  <p className="text-xs uppercase tracking-[0.28em] text-white/45">相关关系</p>
                  <div className="mt-4 space-y-3">
                    {selected.related_concepts.slice(0, 10).map((relation, index) => (
                      <div key={`${relation.concept_id}-${relation.target_name}-${relation.relation_type}-${index}`} className="rounded-[18px] border border-white/8 px-4 py-3">
                        <p className="text-sm font-medium text-white">{relation.target_name}</p>
                        <p className="mt-1 text-xs uppercase tracking-[0.24em] text-white/48">{relation.relation_type}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </>
        ) : (
          <div className="grid min-h-[500px] place-items-center text-sm text-white/58">没有可显示的概念数据。</div>
        )}
      </motion.section>
    </div>
  );
}
