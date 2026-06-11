"use client";

import dynamic from "next/dynamic";
import { forwardRef, useCallback, useEffect, useImperativeHandle, useMemo, useRef } from "react";
import type { GraphResponse } from "@course-kg/shared";
import type { ECharts, EChartsOption, SetOptionOpts } from "echarts";
import type { ComponentType } from "react";
import type { EChartsReactProps } from "echarts-for-react";

const ReactECharts = dynamic(
  () => import("echarts-for-react").then((module) => module.default as unknown as ComponentType<EChartsReactProps>),
  { ssr: false },
);

const palette: Record<string, string> = {
  knowledge_base: "#a5e9ff",
  partition: "#8f97ff",
  document: "#6be2bf",
  section: "#7dd3fc",
  active_chunk: "#94a3b8",
  evidence_atom: "#fbbf24",
  signal_node: "#c084fc",
  evidence_chunk: "#fbbf24",
  document_version: "#6be2bf",
  topic: "#63cbff",
  observation: "#a78bfa",
  claim: "#fb7185",
  method: "#5eead4",
  formula: "#facc15",
  metric: "#fb7185",
  algorithm: "#60a5fa",
  definition: "#c084fc",
  theorem: "#f59e0b",
};
const communityPalette = [
  "#5eead4",
  "#60a5fa",
  "#f59e0b",
  "#f472b6",
  "#a78bfa",
  "#34d399",
  "#fb7185",
  "#facc15",
  "#38bdf8",
  "#c084fc",
  "#a3e635",
  "#fdba74",
];

function colorForNode(node: GraphResponse["nodes"][number]): string {
  if (node.category === "signal_node" && typeof node.community_louvain === "number") {
    return communityPalette[Math.abs(node.community_louvain) % communityPalette.length];
  }
  if (node.category === "signal_node" && node.entity_type) {
    return palette[node.entity_type] ?? palette.signal_node;
  }
  return palette[node.category] ?? "#63cbff";
}

function symbolSizeForNode(node: GraphResponse["nodes"][number]): number {
  if (node.category === "signal_node") {
    return 14 + Math.min(26, Math.max((node.value ?? 2) * 0.75, (node.centrality_score ?? 0) * 48, (node.graph_rank_score ?? 0) * 34));
  }
  if (node.category === "evidence_chunk" || node.category === "active_chunk" || node.category === "evidence_atom") {
    return 10 + Math.min(10, (node.value ?? 1) * 2);
  }
  if (node.category === "document" || node.category === "document_version") {
    return 16;
  }
  return 20;
}

export type NetworkCanvasHandle = {
  resetView: () => void;
  fitView: () => void;
  toggleLayoutLock: () => boolean;
};

type NetworkCanvasProps = {
  graph: GraphResponse;
  height?: number | string;
  selectedNodeId?: string | null;
  onNodeClick?: (nodeId: string, category: string) => void;
  onNodeDoubleClick?: (nodeId: string, category: string) => void;
};

type NodePosition = readonly [number, number];
type NodePositionMap = Map<string, NodePosition>;

type RuntimeForceLayout = {
  warmUp: () => void;
  setFixed: (idx: number) => void;
  setUnfixed: (idx: number) => void;
};

type RuntimeSeriesData = {
  getItemLayout?: (dataIndex: number) => unknown;
  setItemLayout?: (dataIndex: number, layout: NodePosition) => void;
};

type RuntimeGraphNode = {
  setLayout: (layout: NodePosition) => void;
};

type RuntimeGraph = {
  getNodeByIndex: (dataIndex: number) => RuntimeGraphNode;
};

type RuntimeSeriesModel = {
  getData?: () => RuntimeSeriesData;
  get?: (path: string | string[]) => unknown;
  getGraph?: () => RuntimeGraph;
  forceLayout?: RuntimeForceLayout | null;
};

type RuntimeGlobalModel = {
  getSeriesByIndex?: (seriesIndex: number) => RuntimeSeriesModel | undefined;
};

type RuntimeGraphView = {
  updateLayout: (seriesModel: RuntimeSeriesModel) => void;
  _layouting?: boolean;
  _layoutTimeout?: ReturnType<typeof setTimeout> | null;
  _startForceLayoutIteration?: (forceLayout: RuntimeForceLayout, api: ECharts, layoutAnimation: boolean) => void;
};

type RuntimeChart = {
  getModel?: () => RuntimeGlobalModel;
  getViewOfSeriesModel?: (seriesModel: RuntimeSeriesModel) => RuntimeGraphView | undefined;
};

export function buildBaseOption(graph: GraphResponse): EChartsOption {
  return {
    animationDuration: 420,
    animationEasing: "cubicOut",
    backgroundColor: "transparent",
    tooltip: {
      backgroundColor: "rgba(4, 10, 28, 0.94)",
      borderColor: "rgba(120, 215, 255, 0.18)",
      textStyle: { color: "#edf6ff" },
    },
    series: [
      {
        type: "graph",
        roam: true,
        roamTrigger: "global",
        left: 0,
        top: 0,
        right: 0,
        bottom: 0,
        layout: "force",
        force: {
          initLayout: "none",
          repulsion: 160,
          edgeLength: [90, 150],
          gravity: 0.03,
          friction: 0.14,
          layoutAnimation: true,
        },
        draggable: true,
        label: {
          show: true,
          color: "#dff7ff",
          fontSize: 11,
          distance: 6,
          overflow: "break",
          width: 140,
        },
        lineStyle: {
          color: "rgba(122, 169, 255, 0.16)",
          width: 1,
          curveness: 0.06,
          opacity: 0.72,
        },
        emphasis: {
          focus: "none",
          scale: 1.04,
          itemStyle: {
            borderWidth: 2,
            borderColor: "rgba(227,248,255,0.95)",
            shadowBlur: 16,
            shadowColor: "rgba(126, 226, 255, 0.18)",
          },
          lineStyle: {
            width: 1.25,
            opacity: 0.78,
          },
          label: {
            color: "#ffffff",
          },
        },
        blur: {
          itemStyle: { opacity: 1 },
          lineStyle: { opacity: 0.72 },
          label: { opacity: 1 },
        },
        edgeLabel: { show: false },
        data: graph.nodes.map((node) => ({
          ...node,
          draggable: true,
          fixed: false,
          symbolSize: symbolSizeForNode(node),
          itemStyle: {
            color: colorForNode(node),
            borderWidth: node.category === "signal_node" && (node.centrality_score ?? 0) > 0.18 ? 1.8 : 0.8,
            borderColor: node.category === "signal_node" && (node.centrality_score ?? 0) > 0.18 ? "rgba(255,255,255,0.72)" : "rgba(255,255,255,0.14)",
            shadowBlur: node.category === "signal_node" ? 10 + Math.min(16, (node.centrality_score ?? 0) * 44) : 7,
            shadowColor: node.category === "signal_node" ? "rgba(255, 255, 255, 0.16)" : "rgba(99, 203, 255, 0.08)",
          },
          label: {
            color: "#dff7ff",
          },
        })),
        links: graph.edges.map((edge) => ({
          source: edge.source,
          target: edge.target,
          relationLabel: edge.label,
          confidence: edge.confidence,
          category: edge.category,
          evidence_chunk_id: edge.evidence_chunk_id,
          lineStyle: {
            color: edge.is_inferred ? "rgba(255, 207, 112, 0.26)" : edge.category === "signal_projection" ? "rgba(216, 180, 254, 0.24)" : edge.category === "semantic" ? "rgba(84, 213, 255, 0.18)" : "rgba(155, 165, 255, 0.11)",
            width: edge.category === "signal_projection" || edge.category === "semantic" ? 0.8 + Math.min(2.8, (edge.weight ?? edge.confidence ?? 0.4) * 2.6) : 0.8,
            opacity: edge.category === "signal_projection" || edge.category === "semantic" ? 0.36 + Math.min(0.48, (edge.weight ?? 0.3) * 0.55) : 0.38,
            type: edge.is_inferred ? "dashed" : "solid",
          },
        })),
      },
    ],
  };
}

export const NetworkCanvas = forwardRef<NetworkCanvasHandle, NetworkCanvasProps>(function NetworkCanvas(
  { graph, height = 620, selectedNodeId = null, onNodeClick, onNodeDoubleClick },
  ref,
) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<ECharts | null>(null);
  const isLockedRef = useRef(false);
  const highlightedNodeRef = useRef<string | null>(null);

  const option = useMemo(() => buildBaseOption(graph), [graph]);

  const setOption = useCallback((nextOption: EChartsOption, opts?: SetOptionOpts) => {
    const instance = chartRef.current;
    if (!instance) {
      return;
    }
    instance.setOption(nextOption, opts);
  }, []);

  const resizeChart = useCallback(() => {
    const instance = chartRef.current;
    const host = hostRef.current;
    if (!instance || !host) {
      return;
    }

    const { width, height: hostHeight } = host.getBoundingClientRect();
    if (width <= 0 || hostHeight <= 0) {
      return;
    }

    instance.resize({ width, height: hostHeight });
  }, []);

  const getCurrentNodePositions = useCallback((): NodePositionMap => {
    const positions: NodePositionMap = new Map();
    const instance = chartRef.current;
    if (!instance) {
      return positions;
    }

    const seriesData = (instance as unknown as RuntimeChart).getModel?.().getSeriesByIndex?.(0)?.getData?.();
    if (!seriesData?.getItemLayout) {
      return positions;
    }

    graph.nodes.forEach((node, index) => {
      const layout = seriesData.getItemLayout?.(index);
      if (!Array.isArray(layout)) {
        return;
      }
      const [x, y] = layout;
      if (typeof x === "number" && typeof y === "number" && Number.isFinite(x) && Number.isFinite(y)) {
        positions.set(node.id, [x, y]);
      }
    });

    return positions;
  }, [graph.nodes]);

  const getRuntimeState = useCallback(() => {
    const instance = chartRef.current;
    const runtimeChart = instance as unknown as RuntimeChart | null;
    const seriesModel = runtimeChart?.getModel?.().getSeriesByIndex?.(0);
    const graphView = seriesModel ? runtimeChart?.getViewOfSeriesModel?.(seriesModel) : undefined;
    const forceLayout = seriesModel?.forceLayout ?? undefined;
    const data = seriesModel?.getData?.();
    const runtimeGraph = seriesModel?.getGraph?.();

    if (!instance || !seriesModel || !graphView || !forceLayout || !data || !runtimeGraph) {
      return null;
    }

    return { instance, seriesModel, graphView, forceLayout, data, runtimeGraph };
  }, []);

  const syncHighlight = useCallback(
    (nodeId: string | null) => {
      const instance = chartRef.current;
      if (!instance) {
        return;
      }

      const previousId = highlightedNodeRef.current;
      if (previousId) {
        const previousIndex = graph.nodes.findIndex((node) => node.id === previousId);
        if (previousIndex >= 0) {
          instance.dispatchAction({ type: "downplay", seriesIndex: 0, dataIndex: previousIndex });
        }
      }

      if (nodeId) {
        const nextIndex = graph.nodes.findIndex((node) => node.id === nodeId);
        if (nextIndex >= 0) {
          instance.dispatchAction({ type: "highlight", seriesIndex: 0, dataIndex: nextIndex });
        }
      }

      highlightedNodeRef.current = nodeId;
    },
    [graph.nodes],
  );

  const resetView = useCallback(() => {
    isLockedRef.current = false;
    setOption(buildBaseOption(graph), { replaceMerge: ["series"] });
    highlightedNodeRef.current = null;
    requestAnimationFrame(() => {
      resizeChart();
      syncHighlight(selectedNodeId);
    });
  }, [graph, resizeChart, selectedNodeId, setOption, syncHighlight]);

  const fitView = useCallback(() => {
    const instance = chartRef.current;
    if (!instance) {
      return;
    }
    resizeChart();
    instance.dispatchAction({ type: "restore" });
    requestAnimationFrame(() => syncHighlight(selectedNodeId));
  }, [resizeChart, selectedNodeId, syncHighlight]);

  const toggleLayoutLock = useCallback(() => {
    const nextLocked = !isLockedRef.current;
    const runtimeState = getRuntimeState();
    if (!runtimeState) {
      return isLockedRef.current;
    }

    const { instance, seriesModel, graphView, forceLayout, data, runtimeGraph } = runtimeState;
    const currentPositions = getCurrentNodePositions();

    graph.nodes.forEach((node, index) => {
      const position = currentPositions.get(node.id);
      if (position) {
        data.setItemLayout?.(index, position);
        runtimeGraph.getNodeByIndex(index).setLayout(position);
      }

      if (nextLocked) {
        forceLayout.setFixed(index);
        return;
      }

      forceLayout.setUnfixed(index);
    });

    if (nextLocked) {
      if (graphView._layoutTimeout) {
        clearTimeout(graphView._layoutTimeout);
      }
      graphView._layoutTimeout = null;
      graphView._layouting = false;
      graphView.updateLayout(seriesModel);
    } else {
      forceLayout.warmUp();
      if (!graphView._layouting && graphView._startForceLayoutIteration) {
        const layoutAnimation = Boolean(seriesModel.get?.(["force", "layoutAnimation"]));
        graphView._startForceLayoutIteration(forceLayout, instance, layoutAnimation);
      }
    }

    isLockedRef.current = nextLocked;
    requestAnimationFrame(() => {
      resizeChart();
      syncHighlight(selectedNodeId);
    });
    return isLockedRef.current;
  }, [getCurrentNodePositions, getRuntimeState, graph, resizeChart, selectedNodeId, syncHighlight]);

  useImperativeHandle(
    ref,
    () => ({
      resetView,
      fitView,
      toggleLayoutLock,
    }),
    [fitView, resetView, toggleLayoutLock],
  );

  useEffect(() => {
    syncHighlight(selectedNodeId);
  }, [selectedNodeId, syncHighlight]);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) {
      return;
    }

    const observer = new ResizeObserver(() => {
      requestAnimationFrame(resizeChart);
    });
    observer.observe(host);

    requestAnimationFrame(resizeChart);

    return () => observer.disconnect();
  }, [resizeChart]);

  useEffect(() => {
    isLockedRef.current = false;
    requestAnimationFrame(() => {
      resizeChart();
      syncHighlight(selectedNodeId);
    });
  }, [graph, height, resizeChart, selectedNodeId, syncHighlight]);

  return (
    <div ref={hostRef} className="min-w-0 overflow-hidden" style={{ height, width: "100%" }}>
      <ReactECharts
        option={option}
        notMerge
        style={{ height: "100%", width: "100%" }}
        onChartReady={(instance) => {
          chartRef.current = instance;
          requestAnimationFrame(() => {
            resizeChart();
            syncHighlight(selectedNodeId);
          });
        }}
        onEvents={{
          click: (params: { data?: { id?: string; category?: string } }) => {
            if (params.data?.id) {
              onNodeClick?.(params.data.id, params.data.category ?? "signal_node");
            }
          },
          dblclick: (params: { data?: { id?: string; category?: string } }) => {
            if (params.data?.id) {
              onNodeDoubleClick?.(params.data.id, params.data.category ?? "signal_node");
            }
          },
        }}
      />
    </div>
  );
});
