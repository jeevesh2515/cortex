/**
 * Live wikilink graph for the sidebar.
 *
 * Why a radial layout, not force-directed? Two reasons specific to a personal
 * vault:
 *
 *   1. Determinism. A force simulation settles slightly differently each run,
 *      which makes a sidebar widget feel "jumpy". Radial-from-hub is identical
 *      every render -- the only thing that changes is which note is the hub.
 *   2. Signal density. The most useful observation on a personal vault is
 *      "what does this hub connect to"; a ringed layout makes that
 *      first-glance obvious in a way an organic mesh does not.
 *
 * The component fetches the ``/api/graph`` snapshot on mount, then re-polls
 * every 60 s. Network blips are silent (the client returns ``null``) so a
 * missing frame does not look like an error -- the graph view is decorative,
 * not authoritative.
 *
 * ``reindexEpoch`` is a number the parent bumps whenever the user reindexes;
 * it is part of the polling effect's dep list so a manual reindex refreshes
 * the graph immediately rather than waiting for the next tick.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import { RefreshCcw, Network } from 'lucide-react';

import { getGraphSnapshot } from '../../api/client';
import type { GraphEdge, GraphNode, GraphSnapshot } from '../../types';

const POLL_INTERVAL_MS = 60_000;
const DEFAULT_LIMIT = 60;

// ViewBox is a fixed-shape canvas we scale responsively. The dimensions are
// chosen so a ring of 60 nodes at radius 200 leaves comfortable padding while
// staying below one full sidebar panel height -- the sidebar already scrolls.
const SVG_WIDTH = 600;
const SVG_HEIGHT = 360;
const CENTER_X = SVG_WIDTH / 2;
const CENTER_Y = SVG_HEIGHT / 2;
// Radii sized so 4 concentric rings (centre, neighbours, 2-hop, 3-hop) all
// fit inside the SVG with margins. Ring spacing widens slightly as the
// radius grows so labels at the outer edge stay readable.
const RING_RADII = [80, 145, 200, 240];
// Visual bounds. Nodes grow with degree but we clamp to keep tiny hubs and
// sprawling megahubs both legible.
const MIN_RADIUS = 4;
const MAX_RADIUS = 14;

export interface GraphViewProps {
  /** The vault's display name -- needed to build ``obsidian://`` URIs. */
  vaultName: string;
  /**
   * Bumps whenever the user reindexes. Used as part of the polling effect's
   * dep list so the panel refetches without the Sidebar's other panels also
   * tearing down on every edit.
   */
  reindexEpoch?: number;
}

export function GraphView({ vaultName, reindexEpoch = 0 }: GraphViewProps) {
  const [snapshot, setSnapshot] = useState<GraphSnapshot | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<boolean>(false);
  const [manualTick, setManualTick] = useState(0);

  const fetchGraph = useCallback(async () => {
    const snap = await getGraphSnapshot({ limit: DEFAULT_LIMIT });
    if (snap === null) {
      setError(true);
      setLoading(false);
      return;
    }
    setSnapshot(snap);
    setError(false);
    setLoading(false);
  }, []);

  // Initial fetch + 60 s polling. The polling cadence is a deliberate
  // trade-off: the graph mutates only on real edits, which are rarer than
  // chats, but the user should *feel* that it is alive. 60 s is cheap (~30
  // bytes of network unless a reindex lands) and short enough that a fresh
  // note shows up before the next time the user glances at the sidebar.
  useEffect(() => {
    void fetchGraph();
    const handle = window.setInterval(() => {
      void fetchGraph();
    }, POLL_INTERVAL_MS);
    return () => {
      window.clearInterval(handle);
    };
  }, [fetchGraph, reindexEpoch, manualTick]);

  const manualRefresh = useCallback(() => {
    setLoading(true);
    setManualTick((t) => t + 1);
  }, []);

  return (
    <section
      aria-label="Wikilink graph"
      className="rounded-2xl border border-ink-800/80 bg-ink-900/55 p-3"
    >
      <header className="mb-2 flex items-center justify-between gap-2">
        <h3 className="inline-flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-[0.18em] text-ink-400">
          <Network size={11} /> Graph · live
        </h3>
        <div className="inline-flex items-center gap-2 font-mono text-[10px] text-ink-500">
          {snapshot && (
            <span>
              {snapshot.stats.linked_notes}/{snapshot.stats.notes} linked ·{' '}
              {snapshot.stats.edges} edges
            </span>
          )}
          <button
            type="button"
            onClick={manualRefresh}
            disabled={loading}
            className="inline-flex h-6 w-6 items-center justify-center rounded-full border border-ink-700/70 text-ink-400 transition hover:border-violet-500/60 hover:text-violet-200 focus-ring disabled:opacity-50"
            aria-label="Refresh graph"
            title="Refresh graph"
          >
            <RefreshCcw
              size={10}
              className={loading ? 'animate-spin' : ''}
            />
          </button>
        </div>
      </header>
      <GraphCanvas
        snapshot={snapshot}
        loading={loading}
        error={error}
        vaultName={vaultName}
      />
      <p className="mt-2 px-1 font-mono text-[10px] uppercase tracking-wide text-ink-600">
        hub-first radial · click a node to open in Obsidian
      </p>
    </section>
  );
}

/* ------------------------------------------------------------------------ */

interface GraphCanvasProps {
  snapshot: GraphSnapshot | null;
  loading: boolean;
  error: boolean;
  vaultName: string;
}

function GraphCanvas({
  snapshot,
  loading,
  error,
  vaultName,
}: GraphCanvasProps) {
  if (loading && !snapshot) {
    return (
      <div className="flex h-[200px] items-center justify-center text-[11px] text-ink-500">
        Loading graph…
      </div>
    );
  }
  if (error && !snapshot) {
    return (
      <div className="flex h-[200px] flex-col items-center justify-center gap-1 text-center text-[11px] text-ink-500">
        <span>Graph not available.</span>
        <span className="text-ink-600">
          Reindex the vault or hit refresh.
        </span>
      </div>
    );
  }
  if (!snapshot || snapshot.nodes.length === 0) {
    return (
      <div className="flex h-[200px] flex-col items-center justify-center gap-1 text-center text-[11px] text-ink-500">
        <span>No links yet.</span>
        <span className="text-ink-600">
          Add a <code className="font-mono">[[wikilink]]</code> between notes
          and reindex.
        </span>
      </div>
    );
  }
  return <RadialGraph snapshot={snapshot} vaultName={vaultName} />;
}

/* ------------------------------------------------------------------------ */

interface RadialNodePos {
  node: GraphNode;
  cx: number;
  cy: number;
  level: number;
}

function RadialGraph({
  snapshot,
  vaultName,
}: {
  snapshot: GraphSnapshot;
  vaultName: string;
}) {
  const positions = useMemo(() => computeLayout(snapshot), [snapshot]);
  const indexById = useMemo(() => {
    const m = new Map<string, RadialNodePos>();
    for (const p of positions) m.set(p.node.id, p);
    return m;
  }, [positions]);

  const maxDegree = Math.max(...snapshot.nodes.map((n) => n.degree), 1);
  const buildUri = useCallback(
    (noteId: string) =>
      `obsidian://open?vault=${encodeURIComponent(vaultName)}&file=${encodeURIComponent(noteId)}`,
    [vaultName],
  );

  const hubPos = snapshot.center ? indexById.get(snapshot.center) : null;

  return (
    <svg
      viewBox={`0 0 ${SVG_WIDTH} ${SVG_HEIGHT}`}
      className="block h-auto w-full select-none"
      role="img"
      aria-label={`Wikilink graph with ${snapshot.nodes.length} notes and ${snapshot.edges.length} connections`}
    >
      {/* Soft halo around the hub for visual hierarchy. A radial gradient
          under the focal point gives it visual weight without needing an
          animation -- so the radial layout reads at a glance. */}
      <defs>
        <radialGradient id="hub-glow" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="#a78bfa" stopOpacity="0.55" />
          <stop offset="100%" stopColor="#a78bfa" stopOpacity="0" />
        </radialGradient>
      </defs>

      {hubPos && (
        <circle
          cx={hubPos.cx}
          cy={hubPos.cy}
          r={28}
          fill="url(#hub-glow)"
          aria-hidden
        />
      )}

      {/* Edges drawn before nodes so dot fills cover the line endpoints. */}
      {snapshot.edges.map((edge) => {
        const src = indexById.get(edge.source);
        const tgt = indexById.get(edge.target);
        if (!src || !tgt) return null;
        return (
          <EdgePath
            key={`${edge.source}->${edge.target}-${edge.kind}`}
            edge={edge}
            src={src}
            tgt={tgt}
          />
        );
      })}

      {positions.map((pos) => (
        <GraphDot
          key={pos.node.id}
          pos={pos}
          maxDegree={maxDegree}
          uri={buildUri(pos.node.id)}
        />
      ))}

      {/* Truncation hint: a small footer line acknowledging we did not show
          every node. Keeps the user from over-trusting what is on screen. */}
      {snapshot.truncated && (
        <text
          x={SVG_WIDTH - 6}
          y={SVG_HEIGHT - 6}
          textAnchor="end"
          className="fill-ink-600 font-mono text-[10px]"
        >
          + more
        </text>
      )}
    </svg>
  );
}

/* ------------------------------------------------------------------------ */

function EdgePath({
  edge,
  src,
  tgt,
}: {
  edge: GraphEdge;
  src: RadialNodePos;
  tgt: RadialNodePos;
}) {
  // Quadratic bezier with a control point biased outward from the canvas
  // middle keeps the chord curvature consistent. Inside the ring reads
  // as a domain, outside reads as a periphery; without the bias the lines
  // across the centre collide into a clump. 18% is enough to bend the
  // line so it does not look ruler-straight.
  const mx = (src.cx + tgt.cx) / 2;
  const my = (src.cy + tgt.cy) / 2;
  const outwardsX = mx + (mx - CENTER_X) * 0.18;
  const outwardsY = my + (my - CENTER_Y) * 0.18;
  const d = `M${src.cx.toFixed(1)},${src.cy.toFixed(1)} Q${outwardsX.toFixed(1)},${outwardsY.toFixed(1)} ${tgt.cx.toFixed(1)},${tgt.cy.toFixed(1)}`;

  if (edge.kind === 'embed') {
    return (
      <path
        d={d}
        stroke="#fb7185"
        strokeOpacity={0.7}
        strokeWidth={1.6}
        fill="none"
        strokeLinecap="round"
        aria-label={`transclusion ${edge.source} embeds ${edge.target}`}
      >
        <title>{`${edge.source} \u2283 ${edge.target}`}</title>
      </path>
    );
  }
  return (
    <path
      d={d}
      stroke="#8b7fd9"
      strokeOpacity={0.45}
      strokeWidth={0.9}
      fill="none"
      strokeLinecap="round"
      aria-label={`link ${edge.source} to ${edge.target}`}
    >
      <title>{`${edge.source} \u2192 ${edge.target}`}</title>
    </path>
  );
}

/* ------------------------------------------------------------------------ */

function GraphDot({
  pos,
  maxDegree,
  uri,
}: {
  pos: RadialNodePos;
  maxDegree: number;
  uri: string;
}) {
  // Sizing rule: linear interpolation between min and max as degree grows.
  // We always keep at least MIN_RADIUS so even a 1-link node reads as a node.
  const scaled =
    maxDegree > 1
      ? MIN_RADIUS +
        ((pos.node.degree - 1) / (maxDegree - 1)) * (MAX_RADIUS - MIN_RADIUS)
      : MIN_RADIUS;
  const r = Math.max(MIN_RADIUS, Math.min(MAX_RADIUS, scaled));
  const fill = pos.node.is_hub ? '#c4b5fd' : '#312e4a';
  const stroke = pos.node.is_hub ? '#a78bfa' : '#6d28d9';
  return (
    <a
      href={uri}
      target="_blank"
      rel="noopener noreferrer"
      aria-label={`Open ${pos.node.title} in Obsidian (${pos.node.id}, ${pos.node.degree} links)`}
    >
      <circle
        cx={pos.cx}
        cy={pos.cy}
        r={r}
        fill={fill}
        stroke={stroke}
        strokeWidth={pos.node.is_hub ? 2 : 1}
        className="cursor-pointer transition-all duration-150 hover:fill-violet-200 hover:stroke-violet-300"
      >
        <title>{`${pos.node.title}\n${pos.node.id} \u00b7 ${pos.node.degree} links`}</title>
      </circle>
      {/* Label only on the hub and very high-degree neighbours so the canvas
          stays readable. The midpoint-of-maxDegree cutoff prevents a single
          big hub from labelling a dozen of its neighbours at once. */}
      {(pos.node.is_hub ||
        pos.node.degree >= Math.max(4, maxDegree / 2)) && (
        <text
          x={pos.cx}
          y={pos.cy + r + 12}
          textAnchor="middle"
          className="pointer-events-none fill-ink-300 font-mono text-[9px]"
        >
          {trimLabel(pos.node.title)}
        </text>
      )}
    </a>
  );
}

function trimLabel(title: string): string {
  if (title.length <= 14) return title;
  return `${title.slice(0, 13)}\u2026`;
}

/* ------------------------------------------------------------------------ */

/**
 * BFS from the centre to a depth map. Used as the ring assignment for the
 * radial layout -- every connected note becomes a level on the snail.
 */
function bfsDistances(snapshot: GraphSnapshot): Map<string, number> {
  const result = new Map<string, number>();
  if (snapshot.nodes.length === 0) return result;
  const adj: Map<string, Set<string>> = new Map();
  for (const n of snapshot.nodes) adj.set(n.id, new Set());
  for (const e of snapshot.edges) {
    if (!adj.has(e.source) || !adj.has(e.target)) continue;
    adj.get(e.source)!.add(e.target);
    adj.get(e.target)!.add(e.source);
  }

  const start = snapshot.center ?? snapshot.nodes[0].id;
  const queue: Array<[string, number]> = [[start, 0]];
  result.set(start, 0);
  while (queue.length) {
    const [node, depth] = queue.shift()!;
    if (depth >= RING_RADII.length - 1) continue;
    for (const n of adj.get(node) ?? []) {
      if (result.has(n)) continue;
      result.set(n, depth + 1);
      queue.push([n, depth + 1]);
    }
  }
  // Anything the BFS did not reach is in a disconnected cluster -- put them
  // on the outermost ring rather than overlapping a connected component.
  for (const n of snapshot.nodes) {
    if (!result.has(n.id)) result.set(n.id, RING_RADII.length);
  }
  return result;
}

function computeLayout(snapshot: GraphSnapshot): RadialNodePos[] {
  const distances = bfsDistances(snapshot);
  const byLevel = new Map<number, GraphNode[]>();
  for (const node of snapshot.nodes) {
    const lvl = distances.get(node.id) ?? 0;
    if (!byLevel.has(lvl)) byLevel.set(lvl, []);
    byLevel.get(lvl)!.push(node);
  }

  // Within each ring, sort by descending degree so the largest nodes get
  // the visually-friendliest positions (top, 3-o'clock, etc). On ties,
  // fall back to alphabetical so the layout is stable across runs.
  const out: RadialNodePos[] = [];
  for (const [level, members] of byLevel) {
    const sorted = [...members].sort(
      (a, b) => b.degree - a.degree || a.id.localeCompare(b.id),
    );
    const radius = RING_RADII[Math.min(level, RING_RADII.length - 1)];
    const n = sorted.length;
    sorted.forEach((node, i) => {
      // Odd levels are slightly rotated so adjacent rings do not align
      // vertically -- that looks like a wall, not a circle.
      const offset = level % 2 === 0 ? 0 : Math.PI / Math.max(n, 1);
      const angle =
        -Math.PI / 2 + offset + (2 * Math.PI * i) / Math.max(n, 1);
      out.push({
        node,
        cx: CENTER_X + radius * Math.cos(angle),
        cy: CENTER_Y + radius * Math.sin(angle),
        level,
      });
    });
  }
  return out;
}
