"use client";

import { createContext, useContext, useEffect, useMemo } from "react";
import type { KnowledgeBaseCreateRequest, KnowledgeBaseSummary } from "@course-kg/shared";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { createKnowledgeBase, fetchKnowledgeBases } from "@/lib/api";
import { useLocalStorage } from "@/hooks/use-local-storage";

type KnowledgeBaseContextValue = {
  knowledgeBases: KnowledgeBaseSummary[];
  selectedKnowledgeBaseId: string | null;
  selectedKnowledgeBase: KnowledgeBaseSummary | null;
  isLoading: boolean;
  error: Error | null;
  setSelectedKnowledgeBaseId: (value: string | null) => void;
  createKnowledgeBaseSpace: (payload: KnowledgeBaseCreateRequest) => Promise<KnowledgeBaseSummary>;
  isCreating: boolean;
};

const KnowledgeBaseContext = createContext<KnowledgeBaseContextValue | null>(null);

export function KnowledgeBaseProvider({ children }: { children: React.ReactNode }) {
  const queryClient = useQueryClient();
  const [selectedKnowledgeBaseId, setSelectedKnowledgeBaseId] = useLocalStorage<string | null>("knowledgeBase.selectedId", null);
  const knowledgeBasesQuery = useQuery({ queryKey: ["knowledgeBases"], queryFn: fetchKnowledgeBases });

  useEffect(() => {
    if (!knowledgeBasesQuery.data) {
      return;
    }
    const knowledgeBases = knowledgeBasesQuery.data;
    if (knowledgeBases.length === 0) {
      if (selectedKnowledgeBaseId !== null) {
        setSelectedKnowledgeBaseId(null);
      }
      return;
    }
    if (!selectedKnowledgeBaseId || !knowledgeBases.some((knowledgeBase) => knowledgeBase.id === selectedKnowledgeBaseId)) {
      setSelectedKnowledgeBaseId(knowledgeBases[0].id);
    }
  }, [knowledgeBasesQuery.data, selectedKnowledgeBaseId, setSelectedKnowledgeBaseId]);

  const createKnowledgeBaseMutation = useMutation({
    mutationFn: (payload: KnowledgeBaseCreateRequest) => createKnowledgeBase(payload),
    onSuccess: async (knowledgeBase) => {
      queryClient.setQueryData<KnowledgeBaseSummary[]>(["knowledgeBases"], (current) => {
        const base = current ?? [];
        return base.some((item) => item.id === knowledgeBase.id) ? base : [...base, knowledgeBase];
      });
      setSelectedKnowledgeBaseId(knowledgeBase.id);
      await queryClient.invalidateQueries({ queryKey: ["knowledgeBases"] });
    },
  });

  const knowledgeBases = useMemo(() => knowledgeBasesQuery.data ?? [], [knowledgeBasesQuery.data]);
  const selectedKnowledgeBase = knowledgeBases.find((knowledgeBase) => knowledgeBase.id === selectedKnowledgeBaseId) ?? null;

  const value = useMemo<KnowledgeBaseContextValue>(
    () => ({
      knowledgeBases,
      selectedKnowledgeBaseId,
      selectedKnowledgeBase,
      isLoading: knowledgeBasesQuery.isLoading,
      error: (knowledgeBasesQuery.error as Error | null) ?? null,
      setSelectedKnowledgeBaseId,
      createKnowledgeBaseSpace: createKnowledgeBaseMutation.mutateAsync,
      isCreating: createKnowledgeBaseMutation.isPending,
    }),
    [
      knowledgeBases,
      selectedKnowledgeBaseId,
      selectedKnowledgeBase,
      knowledgeBasesQuery.isLoading,
      knowledgeBasesQuery.error,
      setSelectedKnowledgeBaseId,
      createKnowledgeBaseMutation.mutateAsync,
      createKnowledgeBaseMutation.isPending,
    ],
  );

  return <KnowledgeBaseContext.Provider value={value}>{children}</KnowledgeBaseContext.Provider>;
}

export function useKnowledgeBaseContext() {
  const context = useContext(KnowledgeBaseContext);
  if (!context) {
    throw new Error("useKnowledgeBaseContext must be used within KnowledgeBaseProvider");
  }
  return context;
}
