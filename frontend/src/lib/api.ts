/** Thin API client for the LocalChat backend. */

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface ChatRequest {
  messages: ChatMessage[];
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
}

export interface ModelInfo {
  name: string;
  size_bytes: number;
}

export interface HealthResponse {
  status: "ok" | "degraded";
  llama_server: boolean;
  llama_server_detail: string;
  model: string | null;
  models_dir: string;
  exists: boolean;
  models: ModelInfo[];
}

const BASE = "/api";

/** Fetch a single JSON response. */
async function get<T>(path: string): Promise<T> {
  const r = await fetch(`${BASE}${path}`);
  if (!r.ok) throw new Error(`${path} failed: ${r.status}`);
  return (await r.json()) as T;
}

/** POST JSON, return parsed JSON. */
async function post<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${path} failed: ${r.status}`);
  return (await r.json()) as T;
}

/** Stream a chat completion, calling `onToken` for each chunk of text. */
export async function streamChat(
  req: ChatRequest,
  onToken: (text: string) => void,
 _signal?: AbortSignal,
): Promise<void> {
  const r = await fetch(`${BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
    signal: _signal,
  });
  if (!r.ok || !r.body) throw new Error(`chat failed: ${r.status}`);

  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by blank lines.
    let idx: number;
    while ((idx = buffer.indexOf("\n\n")) >= 0) {
      const frame = buffer.slice(0, idx).trim();
      buffer = buffer.slice(idx + 2);
      for (const line of frame.split("\n")) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data: ")) continue;
        const data = trimmed.slice(6);
        if (data === "[DONE]") return;
        try {
          const chunk = JSON.parse(data) as {
            choices?: Array<{ delta?: { content?: string } }>;
          };
          const text = chunk.choices?.[0]?.delta?.content ?? "";
          if (text) onToken(text);
        } catch {
          // Not JSON — ignore (keep the stream alive).
        }
      }
    }
  }
}

export const api = {
  health: () => get<HealthResponse>("/health"),
  models: () => get<{ models: ModelInfo[]; exists: boolean }>("/models"),
  streamChat,
  post,
};