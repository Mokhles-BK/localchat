/** Thin typed client for the LocalChat backend. */

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface Conversation {
  id: number;
  title: string;
  system_prompt: string;
  temperature: number;
  top_p: number;
  max_tokens: number; // -1 = automatic
  created_at: number;
  msg_count?: number;
  last_active?: number;
}

/** The per-conversation fields the user can edit. */
export type ConversationSettings = Pick<
  Conversation,
  "system_prompt" | "temperature" | "top_p" | "max_tokens"
>;
export type ConversationPatch = Partial<ConversationSettings & Pick<Conversation, "title">>;

export interface ModelInfo {
  name: string;
  size_bytes: number;
}

export interface HealthResponse {
  status: string;
  llama_server: boolean;
  llama_server_detail: string;
  model: string | null;
  models_dir: string;
  exists: boolean;
  models: ModelInfo[];
}

export interface StreamRequest {
  conversation_id: number;
  content?: string;
  regenerate?: boolean;
}

const BASE = "/api";

/** Turn a failed response into an Error with the backend's `detail` message. */
async function failure(r: Response, path: string): Promise<Error> {
  let detail = `${r.status}`;
  try {
    const body = await r.json();
    if (body?.detail) {
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    }
  } catch {
    // body was not JSON — keep the status code
  }
  return new Error(`${path}: ${detail}`);
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const r = await fetch(`${BASE}${path}`, {
    method,
    headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) throw await failure(r, path);
  if (r.status === 204) return undefined as T;
  return (await r.json()) as T;
}

/**
 * Stream a reply, calling `onToken` for each chunk of text.
 * Aborting `signal` closes the connection; the backend then saves whatever
 * was generated so far, so the database stays the source of truth.
 */
export async function streamChat(
  req: StreamRequest,
  onToken: (text: string) => void,
  signal?: AbortSignal,
): Promise<void> {
  const r = await fetch(`${BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
    signal,
  });
  if (!r.ok || !r.body) throw await failure(r, "/chat");

  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by a blank line.
    let idx: number;
    while ((idx = buffer.indexOf("\n\n")) >= 0) {
      const frame = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      for (const line of frame.split("\n")) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data: ")) continue;
        const data = trimmed.slice(6);
        if (data === "[DONE]") return;
        try {
          const chunk = JSON.parse(data) as {
            error?: string;
            choices?: Array<{ delta?: { content?: string } }>;
          };
          if (chunk.error) throw new Error(chunk.error);
          const text = chunk.choices?.[0]?.delta?.content ?? "";
          if (text) onToken(text);
        } catch (e) {
          // Re-throw backend error frames; ignore non-JSON noise.
          if (e instanceof SyntaxError) continue;
          throw e;
        }
      }
    }
  }
}

export const api = {
  health: () => request<HealthResponse>("GET", "/health"),
  listConversations: async () =>
    (await request<{ conversations: Conversation[] }>("GET", "/conversations")).conversations,
  createConversation: (body: Partial<ConversationPatch> = {}) =>
    request<Conversation>("POST", "/conversations", body),
  updateConversation: (id: number, patch: ConversationPatch) =>
    request<Conversation>("PATCH", `/conversations/${id}`, patch),
  deleteConversation: (id: number) => request<void>("DELETE", `/conversations/${id}`),
  messages: async (id: number) =>
    (await request<{ messages: ChatMessage[] }>("GET", `/conversations/${id}/messages`)).messages,
  streamChat,
};
