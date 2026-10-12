/** Thin typed client for the LocalChat backend. */

/** Token counts and speed saved with each assistant reply. */
export interface MessageMeta {
  tokens: number;
  prompt_tokens: number;
  ctx: number;
  exact: boolean;
  tps?: number;
}

export interface Attachment {
  id: string;
  name: string;
  mime: string;
  kind: "image" | "text";
  size: number;
}

export interface ChatMessage {
  attachments?: Attachment[];
  id?: number; // missing only while a message is still being created
  role: "system" | "user" | "assistant";
  content: string;
  meta?: MessageMeta | null;
}

export interface SearchHit {
  conversation_id: number;
  title: string;
  message_id: number | null; // null = the title matched
  role: "user" | "assistant" | null;
  snippet: string;
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

export type Fit = "gpu" | "partial" | "too_big" | "unknown";

export interface ModelInfo {
  name: string;
  size_bytes: number;
  quant?: string | null;
  fit?: Fit;
  projector?: string | null; // vision projector loaded with this model, if any
}

export interface Projector {
  name: string;
  size_bytes: number;
}

export interface RepoResult {
  repo_id: string;
  downloads: number;
  likes: number;
}

export interface RepoFile {
  kind: "model" | "projector";
  filename: string;
  size_bytes: number;
  quant: string | null;
  fit: Fit;
  installed: boolean;
  downloadable: boolean;
  note: string | null;
}

export interface LocalCandidate {
  kind: "model" | "projector";
  path: string;
  name: string;
  size_bytes: number;
  source: "ollama" | "lmstudio" | "folder";
  quant: string | null;
  fit: Fit;
  installed: boolean;
}

export interface DownloadJob {
  id: string;
  repo_id: string;
  filename: string;
  total: number;
  downloaded: number;
  status: "queued" | "downloading" | "done" | "error" | "cancelled";
  error: string | null;
}

export function formatBytes(n: number): string {
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(2)} GB`;
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(0)} MB`;
  return `${Math.max(1, Math.round(n / 1024))} KB`;
}

export interface HealthResponse {
  status: string;
  llama_server: boolean;
  llama_server_detail: string;
  model: string | null;
  loading_model: string | null;
  ctx_size: number;
  vision: boolean;
  models_dir: string;
  exists: boolean;
  models: ModelInfo[];
}

export interface StreamRequest {
  conversation_id: number;
  content?: string;
  regenerate?: boolean;
  edit_message_id?: number;
  attachments?: string[];
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
  search: async (q: string) =>
    (await request<{ results: SearchHit[] }>("GET", `/search?q=${encodeURIComponent(q)}`)).results,
  forkConversation: (id: number, upToMessageId: number) =>
    request<Conversation>("POST", `/conversations/${id}/fork`, { up_to_message_id: upToMessageId }),
  listModels: async () => {
    const r = await request<{ models: ModelInfo[]; projectors: Projector[] }>("GET", "/models");
    return { models: r.models, projectors: r.projectors ?? [] };
  },
  pairModel: (model: string, projector: string) =>
    request<{ model: string; projector: string | null; restart_needed: boolean }>("POST", "/models/pair", {
      model,
      projector,
    }),
  uploadFile,
  searchRepos: async (q: string) =>
    (await request<{ results: RepoResult[] }>("GET", `/models/search?q=${encodeURIComponent(q)}`)).results,
  repoFiles: async (repoId: string) =>
    (await request<{ files: RepoFile[] }>("GET", `/models/files?repo_id=${encodeURIComponent(repoId)}`)).files,
  scan: async (folder?: string) =>
    (
      await request<{ candidates: LocalCandidate[] }>(
        "GET",
        `/models/scan${folder ? `?folder=${encodeURIComponent(folder)}` : ""}`,
      )
    ).candidates,
  importModel: (path: string, name?: string) =>
    request<DownloadJob>("POST", "/models/import", { path, name }),
  startDownload: (repo_id: string, filename: string) =>
    request<DownloadJob>("POST", "/models/download", { repo_id, filename }),
  downloads: async () =>
    (await request<{ downloads: DownloadJob[] }>("GET", "/models/downloads")).downloads,
  cancelDownload: (id: string) => request<void>("DELETE", `/models/downloads/${id}`),
  activateModel: (name: string) =>
    request<{ model: string; vision: boolean; warning: string | null }>("POST", "/models/activate", { name }),
  deleteModel: (name: string) => request<void>("DELETE", `/models/${encodeURIComponent(name)}`),
};

/** Download link for a conversation export (the browser handles the file). */
export const exportUrl = (id: number, format: "md" | "json") =>
  `${BASE}/conversations/${id}/export?format=${format}`;

export const attachmentUrl = (id: string) => `${BASE}/uploads/${id}`;

const MAX_IMAGE_SIDE = 1280;

/** Shrink big photos before upload: smaller upload, fewer image tokens, faster. */
async function prepareImage(file: File): Promise<{ blob: Blob; name: string }> {
  if (!file.type.startsWith("image/") || file.type === "image/gif") return { blob: file, name: file.name };
  try {
    const bmp = await createImageBitmap(file);
    const scale = Math.min(1, MAX_IMAGE_SIDE / Math.max(bmp.width, bmp.height));
    if (scale === 1 && file.size < 1_500_000) {
      bmp.close();
      return { blob: file, name: file.name };
    }
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(bmp.width * scale);
    canvas.height = Math.round(bmp.height * scale);
    const ctx = canvas.getContext("2d");
    if (!ctx) return { blob: file, name: file.name };
    ctx.fillStyle = "#fff"; // JPEG has no transparency
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(bmp, 0, 0, canvas.width, canvas.height);
    bmp.close();
    const blob = await new Promise<Blob | null>((r) => canvas.toBlob(r, "image/jpeg", 0.85));
    if (!blob) return { blob: file, name: file.name };
    return { blob, name: file.name.replace(/\.[^.]+$/, "") + ".jpg" };
  } catch {
    return { blob: file, name: file.name };
  }
}

function toBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",")[1] ?? "");
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });
}

/** Upload an image or text/code file; returns the attachment to send with a message. */
export async function uploadFile(file: File): Promise<Attachment> {
  const { blob, name } = await prepareImage(file);
  return request<Attachment>("POST", "/uploads", { name, data_base64: await toBase64(blob) });
}
