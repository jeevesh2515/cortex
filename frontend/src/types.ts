/**
 * Cortex frontend types -- shared with the FastAPI wire format.
 *
 * Keep this file in lockstep with `src/cortex/server/schemas.py`. The names
 * are deliberately aligned so a reader moving between server and client
 * sees ``CitationOut``->``Citation`` and not two different shapes for the
 * same concept.
 */

export type ChatRole = 'user' | 'assistant' | 'system';

export interface ChatMessage {
  role: ChatRole;
  content: string;
  ts?: string;
}

export interface Citation {
  index: number;
  note_id: string;
  obsidian_uri: string;
  title: string;
  snippet: string;
  score: number;
  tags?: string[];
}

export interface ProviderStatus {
  name: string;
  model: string;
  policy: string;
  priority: number;
  eligible_private: boolean;
  reason?: string | null;
  config_issue?: string | null;
}

export interface ThermalStatus {
  state: string;
  power?: string | null;
  cpu_speed_limit?: number | null;
  battery_percent?: number | null;
  workers: number;
  may_backfill: boolean;
  available: boolean;
  reason?: string | null;
}

export interface StatusSnapshot {
  vault_path: string;
  vault_name: string;
  notes_indexed: number;
  chunks: number;
  providers: ProviderStatus[];
  thermal: ThermalStatus;
  memory_count: number;
  memory_enabled: boolean;
}

export interface MemoryNoteOut {
  path: string;
  title: string;
  created: string;
  question: string;
  answer: string;
  sources: string[];
  obsidian_uri: string;
}

export interface WhoAmI {
  vault_name: string;
  vault_path: string;
  providers: Array<{
    name: string;
    model: string;
    policy: string;
    configured: boolean;
  }>;
  memory_enabled: boolean;
  local_only: boolean;
}

// SSE event variants emitted by ``POST /api/chat``. Plain discriminated union
// so an ``EventSource``'s listener gets ``type`` for free.
export type ChatEvent =
  | { type: 'provider'; data: { deciding?: boolean; name?: string; model?: string; escalated?: boolean; policy?: string; elapsed_ms?: number; local_only_enforced?: boolean } }
  | { type: 'retrieval'; data: { query: string; elapsed_ms: number; retrievers: Record<string, number>; date_filter: string | null; reranked: boolean; matched: number } }
  | { type: 'citation'; data: Citation }
  | { type: 'text'; data: { delta: string } }
  | { type: 'memory'; data: { saved: string; obsidian_uri: string } }
  | { type: 'done'; data: { provider: string; model?: string; escalated?: boolean; policy?: string; elapsed_ms?: number; answer: string; total_elapsed_ms?: number } }
  | { type: 'error'; data: { type?: string; message: string } };

export interface ChatRequest {
  messages: ChatMessage[];
  local_only?: boolean;
  remember?: boolean;
  top_k?: number;
}

// -- Wikilink graph -------------------------------------------------------

export interface GraphNode {
  id: string;
  title: string;
  degree: number;
  /** Top-level tag, reserved for a follow-up that carries note-level tags. */
  tag?: string | null;
  is_hub: boolean;
}

export type GraphEdgeKind = 'link' | 'embed';

export interface GraphEdge {
  source: string;
  target: string;
  kind: GraphEdgeKind;
}

export interface GraphSnapshot {
  stats: { notes: number; linked_notes: number; edges: number; embeds: number; tags: number };
  nodes: GraphNode[];
  edges: GraphEdge[];
  center: string | null;
  truncated: boolean;
}
