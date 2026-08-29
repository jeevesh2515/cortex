/**
 * Cortex API client. Single source of truth on the wire; the rest of the UI
 * imports names from here and never touches ``fetch`` directly.
 *
 * ``openChatStream`` parses Server-Sent Events by reading the ``fetch``
 * response body as a stream and emitting one discriminated ``ChatEvent`` per
 * ``event: <type>\ndata: <json>\n\n`` block. We do not use ``EventSource``
 * because EventSource only supports GET; the chat endpoint is POST so a
 * manual SSE reader is the only path.
 */

import type {
  ChatEvent,
  ChatRequest,
  Citation,
  GraphSnapshot,
  MemoryNoteOut,
  StatusSnapshot,
  WhoAmI,
} from '../types';

// Relative URLs in production (``cortex serve`` mounts /api under /). During
// Vite dev, the proxy on /api in ``vite.config.ts`` forwards to the FastAPI
// port. We never set a base URL on the client, so the same bundle works for
// every deployment.
const BASE = '';

/** Treat a fetch Response as JSON or throw a helpful error. */
async function jsonOrThrow<T>(r: Response): Promise<T> {
  if (!r.ok) {
    let msg = `HTTP ${r.status}`;
    try {
      const body = (await r.json()) as { message?: string; detail?: unknown };
      msg = body.message || body.detail?.toString() || msg;
    } catch {
      // ignore -- non-JSON error bodies stay as the status-line message
    }
    throw new Error(msg);
  }
  return (await r.json()) as T;
}

export async function getWhoAmI(): Promise<WhoAmI> {
  return jsonOrThrow(await fetch(`${BASE}/api/whoami`));
}

export async function getStatus(): Promise<StatusSnapshot> {
  return jsonOrThrow(await fetch(`${BASE}/api/status`));
}

/**
 * Fetch a curated subgraph sized for the sidebar's radial view. ``center``
 * forces a one-hop BFS around the given note; without one, returns the
 * top-degree nodes so the radial layout can show overall vault structure.
 *
 * Returns ``null`` on transient network failure so the polling loop can
 * quietly retry without spamming toast errors -- the graph is decorative,
 * losing one frame is acceptable.
 */
export async function getGraphSnapshot(
  opts: { limit?: number; center?: string | null } = {},
): Promise<GraphSnapshot | null> {
  const params = new URLSearchParams();
  if (opts.limit) params.set('limit', String(opts.limit));
  if (opts.center) params.set('center', opts.center);
  const qs = params.toString();
  try {
    const r = await fetch(`${BASE}/api/graph${qs ? `?${qs}` : ''}`);
    if (!r.ok) {
      // 422 (bad request), 503 (runtime gone) etc -- not transient.
      return null;
    }
    return (await r.json()) as GraphSnapshot;
  } catch {
    // network blip
    return null;
  }
}

export async function reindex(opts: { full?: boolean; ignore_thermal?: boolean } = {}): Promise<unknown> {
  return jsonOrThrow(
    await fetch(`${BASE}/api/reindex`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ full: !!opts.full, ignore_thermal: !!opts.ignore_thermal }),
    }),
  );
}

export async function searchCitations(q: string, topK = 8): Promise<{
  query: string;
  elapsed_ms: number;
  date_filter: string | null;
  retrievers: Record<string, number>;
  results: Citation[];
}> {
  return jsonOrThrow(
    await fetch(`${BASE}/api/search`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ query: q, top_k: topK, use_graph: true }),
    }),
  );
}

export async function listMemoryNotes(limit = 12): Promise<MemoryNoteOut[]> {
  const r = await fetch(`${BASE}/api/memory/recent?limit=${limit}`);
  if (!r.ok) return [];
  return (await r.json()) as MemoryNoteOut[];
}

export async function rememberExchange(req: {
  question: string;
  answer: string;
  sources?: string[];
  tags?: string[];
  provider?: string;
}): Promise<{ saved: string; obsidian_uri: string }> {
  return jsonOrThrow(
    await fetch(`${BASE}/api/memory`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        question: req.question,
        answer: req.answer,
        sources: req.sources ?? [],
        tags: req.tags ?? [],
        provider: req.provider ?? '',
      }),
    }),
  );
}

/**
 * Open an SSE chat stream. Returns an ``AsyncIterable<ChatEvent>`` so callers
 * ``for await`` events naturally and avoid callback soup. Aborts on
 * ``AbortSignal`` so React's ``useEffect`` cleanup can cancel in-flight
 * requests.
 */
export async function* openChatStream(
  req: ChatRequest,
  signal?: AbortSignal,
): AsyncGenerator<ChatEvent> {
  const response = await fetch(`${BASE}/api/chat`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', accept: 'text/event-stream' },
    body: JSON.stringify(req),
    signal,
  });
  if (!response.ok || !response.body) {
    throw new Error(`HTTP ${response.status} on /api/chat`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    // SSE delimiter is ``\n\n`` between events. We loop, append, and split on
    // a blank line. A single ``fetch`` response can carry tens of events; we
    // pull them all out of one read so a slow chunk boundary does not stall
    // the chat.
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let sepAt: number;
      while ((sepAt = buffer.indexOf('\n\n')) !== -1) {
        const raw = buffer.slice(0, sepAt);
        buffer = buffer.slice(sepAt + 2);
        const event = parseSseEvent(raw);
        if (event) yield event;
      }
    }
    // Drain anything left in the buffer when the stream ends.
    if (buffer.trim().length) {
      const tail = parseSseEvent(buffer);
      if (tail) yield tail;
    }
  } finally {
    try {
      reader.releaseLock();
    } catch {
      // already released -- ignore
    }
  }
}

function parseSseEvent(raw: string): ChatEvent | null {
  let eventName: string | null = null;
  const dataLines: string[] = [];
  for (const line of raw.split('\n')) {
    if (line.startsWith('event:')) {
      eventName = line.slice(6).trim();
    } else if (line.startsWith('data:')) {
      dataLines.push(line.slice(5).trim());
    }
    // ``:`` lines are SSE comments (heartbeats); ignored.
  }
  if (!eventName) return null;
  let parsed: Record<string, unknown> = {};
  const joined = dataLines.join('\n');
  if (joined) {
    try {
      parsed = JSON.parse(joined) as Record<string, unknown>;
    } catch {
      parsed = { message: joined };
    }
  }
  return { type: eventName as ChatEvent['type'], data: parsed } as ChatEvent;
}
