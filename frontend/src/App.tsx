import { useEffect, useRef, useState } from "react";
import {
  api,
  type Attachment,
  type ChatMessage,
  type Conversation,
  type ConversationPatch,
  type ConversationSettings,
  exportUrl,
} from "./lib/api";
import Sidebar from "./components/Sidebar";
import SettingsPanel from "./components/SettingsPanel";
import Message from "./components/Message";
import ModelsPanel from "./components/ModelsPanel";

const ACTIVE_KEY = "localchat.activeId";
const DEFAULT_SETTINGS: ConversationSettings = {
  system_prompt: "",
  temperature: 0.7,
  top_p: 0.9,
  max_tokens: -1,
};

interface PendingFile {
  key: string;
  name: string;
  status: "uploading" | "ready" | "error";
  attachment?: Attachment;
  preview?: string; // object URL for image thumbnails
  error?: string;
}

const MAX_ATTACHMENTS = 4;
const ACCEPT =
  "image/png,image/jpeg,image/webp,image/gif,text/*,.md,.py,.js,.jsx,.ts,.tsx,.json,.csv,.html,.css,.java,.c,.h,.cpp,.cs,.go,.rs,.sql,.sh,.ps1,.yml,.yaml,.xml,.log,.toml,.ini";

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export default function App() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveIdState] = useState<number | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState<ConversationSettings>(DEFAULT_SETTINGS);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [model, setModel] = useState<string | null>(null);
  const [loadingModel, setLoadingModel] = useState<string | null>(null);
  const [modelsOpen, setModelsOpen] = useState(false);
  const [ctxSize, setCtxSize] = useState<number | null>(null);
  const [vision, setVision] = useState(false);
  const [pending, setPending] = useState<PendingFile[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const listRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  // The active id as a ref so async callbacks never see a stale value.
  const activeIdRef = useRef<number | null>(null);
  const pendingPatches = useRef<Record<number, ConversationPatch>>({});
  const patchTimer = useRef<number | undefined>(undefined);

  const activeConv = conversations.find((c) => c.id === activeId) ?? null;
  const settings: ConversationSettings = activeConv ?? draft;

  function setActive(id: number | null) {
    activeIdRef.current = id;
    setActiveIdState(id);
    if (id === null) localStorage.removeItem(ACTIVE_KEY);
    else localStorage.setItem(ACTIVE_KEY, String(id));
  }

  // Auto-scroll as tokens stream in.
  useEffect(() => {
    const el = listRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, busy]);

  // Grow the composer with its content.
  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 192)}px`;
  }, [input]);

  // Initial load: health, conversation list, last open conversation.
  useEffect(() => {
    void refreshHealth();
    api
      .listConversations()
      .then(async (list) => {
        setConversations(list);
        const saved = Number(localStorage.getItem(ACTIVE_KEY));
        if (list.some((c) => c.id === saved)) await openConversation(saved);
      })
      .catch((e) => setError((e as Error).message));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function refreshHealth() {
    try {
      const h = await api.health();
      setModel(h.model);
      setLoadingModel(h.loading_model);
      setCtxSize(h.ctx_size ?? null);
      setVision(!!h.vision);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  // While a model is loading (even after a page reload), keep checking.
  useEffect(() => {
    if (!loadingModel) return;
    const id = window.setInterval(() => void refreshHealth(), 2000);
    return () => window.clearInterval(id);
  }, [loadingModel]);

  async function refreshList() {
    try {
      setConversations(await api.listConversations());
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function openConversation(id: number | null) {
    abortRef.current?.abort(); // the backend keeps whatever was generated
    setActive(id);
    setError(null);
    if (id === null) {
      setMessages([]);
      return;
    }
    try {
      const msgs = await api.messages(id);
      if (activeIdRef.current === id) setMessages(msgs);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  /** Stream a reply into conversation `convId`, then re-sync with the database. */
  async function runStream(
    convId: number,
    body: { content?: string; regenerate?: boolean; edit_message_id?: number; attachments?: string[] },
  ) {
    const controller = new AbortController();
    abortRef.current = controller;
    setBusy(true);
    let aborted = false;
    try {
      await api.streamChat(
        { conversation_id: convId, ...body },
        (token) => {
          if (activeIdRef.current !== convId) return;
          setMessages((m) => {
            const copy = [...m];
            const last = copy[copy.length - 1];
            if (last && last.role === "assistant") {
              copy[copy.length - 1] = { ...last, content: last.content + token };
            }
            return copy;
          });
        },
        controller.signal,
      );
    } catch (e) {
      if ((e as Error).name === "AbortError") aborted = true;
      else setError((e as Error).message);
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null;
        setBusy(false);
      }
    }
    // After Stop the backend needs a moment to save the partial reply.
    if (aborted) await sleep(500);
    await refreshList();
    if (activeIdRef.current === convId) {
      try {
        setMessages(await api.messages(convId));
      } catch (e) {
        setError((e as Error).message);
      }
    }
    inputRef.current?.focus();
  }

  /** Upload picked / pasted / dropped files and show them as chips above the composer. */
  async function addFiles(files: File[]) {
    setError(null);
    for (const file of files) {
      if (pending.length + 1 > MAX_ATTACHMENTS) {
        setError(`At most ${MAX_ATTACHMENTS} attachments per message.`);
        break;
      }
      if (file.type.startsWith("image/") && !vision) {
        setError("The loaded model cannot read images. Open the model menu and load a vision model first.");
        continue;
      }
      const key = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
      const preview = file.type.startsWith("image/") ? URL.createObjectURL(file) : undefined;
      setPending((p) => [...p, { key, name: file.name || "pasted image", status: "uploading", preview }]);
      try {
        const attachment = await api.uploadFile(file);
        setPending((p) => p.map((x) => (x.key === key ? { ...x, status: "ready", attachment } : x)));
      } catch (e) {
        setPending((p) => p.map((x) => (x.key === key ? { ...x, status: "error", error: (e as Error).message } : x)));
      }
    }
  }

  function removePending(key: string) {
    setPending((p) => {
      const gone = p.find((x) => x.key === key);
      if (gone?.preview) URL.revokeObjectURL(gone.preview);
      return p.filter((x) => x.key !== key);
    });
  }

  function onPaste(e: React.ClipboardEvent<HTMLTextAreaElement>) {
    const files = Array.from(e.clipboardData.files);
    if (files.length > 0) {
      e.preventDefault();
      void addFiles(files);
    }
  }

  async function send() {
    const text = input.trim();
    const ready = pending.filter((x) => x.status === "ready" && x.attachment).map((x) => x.attachment!);
    const uploading = pending.some((x) => x.status === "uploading");
    if ((!text && ready.length === 0) || uploading || busy || loadingModel) return;
    setError(null);

    let convId = activeIdRef.current;
    if (convId === null) {
      try {
        const conv = await api.createConversation(draft);
        convId = conv.id;
        setConversations((c) => [conv, ...c]);
        setActive(conv.id);
      } catch (e) {
        setError((e as Error).message);
        return;
      }
    }
    setInput("");
    pending.forEach((x) => x.preview && URL.revokeObjectURL(x.preview));
    setPending([]);
    setMessages((m) => [
      ...m,
      { role: "user", content: text, attachments: ready },
      { role: "assistant", content: "" },
    ]);
    await runStream(convId, { content: text, attachments: ready.map((a) => a.id) });
  }

  async function regenerate() {
    const convId = activeIdRef.current;
    if (convId === null || busy) return;
    setError(null);
    setMessages((m) => [...m.slice(0, -1), { role: "assistant", content: "" }]);
    await runStream(convId, { regenerate: true });
  }

  /** Replace a user message, drop everything after it, and re-answer. */
  async function editMessage(index: number, content: string) {
    const convId = activeIdRef.current;
    const target = messages[index];
    if (convId === null || busy || target?.id === undefined) return;
    setError(null);
    setMessages((m) => [
      ...m.slice(0, index),
      { ...m[index], content, meta: null },
      { role: "assistant", content: "" },
    ]);
    await runStream(convId, { content, edit_message_id: target.id });
  }

  /** Copy the conversation up to a message into a new chat and open it. */
  async function forkFrom(index: number) {
    const convId = activeIdRef.current;
    const target = messages[index];
    if (convId === null || busy || target?.id === undefined) return;
    try {
      const conv = await api.forkConversation(convId, target.id);
      await refreshList();
      await openConversation(conv.id);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  function stop() {
    abortRef.current?.abort();
  }

  async function rename(id: number, title: string) {
    try {
      const conv = await api.updateConversation(id, { title });
      setConversations((cs) => cs.map((c) => (c.id === id ? { ...c, title: conv.title } : c)));
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function remove(id: number) {
    const conv = conversations.find((c) => c.id === id);
    if (!window.confirm(`Delete "${conv?.title || "New chat"}"? This cannot be undone.`)) return;
    try {
      await api.deleteConversation(id);
      const rest = conversations.filter((c) => c.id !== id);
      setConversations(rest);
      if (activeIdRef.current === id) await openConversation(rest[0]?.id ?? null);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  /** Settings edits apply instantly in the UI and are saved after a short pause. */
  function changeSettings(patch: Partial<ConversationSettings>) {
    const id = activeIdRef.current;
    if (id === null) {
      setDraft((d) => ({ ...d, ...patch }));
      return;
    }
    setConversations((cs) => cs.map((c) => (c.id === id ? { ...c, ...patch } : c)));
    pendingPatches.current[id] = { ...pendingPatches.current[id], ...patch };
    window.clearTimeout(patchTimer.current);
    patchTimer.current = window.setTimeout(async () => {
      const pending = pendingPatches.current;
      pendingPatches.current = {};
      for (const [cid, p] of Object.entries(pending)) {
        try {
          await api.updateConversation(Number(cid), p);
        } catch (e) {
          setError((e as Error).message);
        }
      }
    }, 500);
  }

  // Context window usage, from the last reply's real prompt + completion tokens.
  const lastMeta = [...messages].reverse().find((m) => m.role === "assistant" && m.meta)?.meta;
  const ctxTotal = lastMeta?.ctx ?? ctxSize;
  const ctxUsed = lastMeta ? lastMeta.prompt_tokens + lastMeta.tokens : null;
  const ctxPct = ctxUsed !== null && ctxTotal ? Math.min(100, (ctxUsed / ctxTotal) * 100) : null;

  function onKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void send();
    }
  }

  return (
    <div className="flex h-screen bg-zinc-950">
      {sidebarOpen && (
        <Sidebar
          conversations={conversations}
          activeId={activeId}
          onSelect={(id) => void openConversation(id)}
          onNew={() => void openConversation(null)}
          onRename={(id, t) => void rename(id, t)}
          onDelete={(id) => void remove(id)}
        />
      )}

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center justify-between border-b border-zinc-800 px-4 py-3">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setSidebarOpen((v) => !v)}
              title="Toggle sidebar"
              className="rounded-md px-2 py-1 text-zinc-400 hover:bg-zinc-900 hover:text-zinc-100"
            >
              ☰
            </button>
            <div className="flex items-center gap-2">
              <div className="h-2.5 w-2.5 rounded-full bg-emerald-500" />
              <h1 className="text-sm font-semibold text-zinc-100">LocalChat</h1>
            </div>
          </div>
          <div className="flex items-center gap-3">
            {ctxPct !== null && ctxTotal && ctxUsed !== null && (
              <span
                title="Context window used by this conversation (older messages are trimmed when it fills up)"
                className={`hidden text-xs sm:inline ${
                  ctxPct > 90 ? "text-red-400" : ctxPct > 75 ? "text-amber-400" : "text-zinc-500"
                }`}
              >
                ctx {ctxUsed.toLocaleString()} / {ctxTotal.toLocaleString()}
              </span>
            )}
            {activeId !== null && (
              <details className="relative">
                <summary className="cursor-pointer list-none rounded-md px-2 py-1 text-xs text-zinc-400 hover:bg-zinc-900 [&::-webkit-details-marker]:hidden">
                  Export ▾
                </summary>
                <div className="absolute right-0 z-20 mt-1 grid w-40 rounded-xl border border-zinc-800 bg-zinc-950 p-1 shadow-xl">
                  {(["md", "json"] as const).map((f) => (
                    <a
                      key={f}
                      href={exportUrl(activeId, f)}
                      download
                      onClick={(e) => e.currentTarget.closest("details")?.removeAttribute("open")}
                      className="rounded-lg px-3 py-1.5 text-xs text-zinc-300 hover:bg-zinc-900"
                    >
                      {f === "md" ? "Markdown (.md)" : "JSON (.json)"}
                    </a>
                  ))}
                </div>
              </details>
            )}
            <button
              onClick={() => setModelsOpen(true)}
              title="Manage models"
              className="max-w-[16rem] truncate rounded-md px-2 py-1 text-xs text-zinc-400 hover:bg-zinc-900 hover:text-zinc-100"
            >
              {loadingModel ? `loading ${loadingModel}…` : (model ?? "no model")}
              {vision && !loadingModel ? " · 👁" : ""} ▾
            </button>
            <button
              onClick={() => setSettingsOpen((v) => !v)}
              className={`rounded-md px-2 py-1 text-xs ${
                settingsOpen ? "bg-zinc-800 text-zinc-100" : "text-zinc-400 hover:bg-zinc-900"
              }`}
            >
              Settings
            </button>
          </div>
        </header>

        {settingsOpen && <SettingsPanel settings={settings} onChange={changeSettings} />}

        <div ref={listRef} className="scroll-thin min-w-0 flex-1 overflow-y-auto px-4 py-6">
          <div className="mx-auto flex w-full min-w-0 max-w-2xl flex-col gap-5">
            {messages.length === 0 && !busy && (
              <div className="py-10 text-center text-zinc-600">
                <p className="text-base text-zinc-400">LocalChat is ready.</p>
                <p className="mt-1 text-sm">Send a message to start a conversation.</p>
              </div>
            )}
            {messages.map((m, i) => (
              <Message
                key={i}
                message={m}
                isLast={i === messages.length - 1}
                streaming={busy}
                onRegenerate={() => void regenerate()}
                onEdit={m.role === "user" && m.id !== undefined && !busy ? (t) => void editMessage(i, t) : undefined}
                onFork={m.id !== undefined && !busy ? () => void forkFrom(i) : undefined}
              />
            ))}
            {error && (
              <div className="rounded-xl border border-red-800/60 bg-red-950/40 px-4 py-3 text-sm text-red-300">
                {error}
              </div>
            )}
          </div>
        </div>

        <div
          className="border-t border-zinc-800 px-4 py-4"
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            void addFiles(Array.from(e.dataTransfer.files));
          }}
        >
          {pending.length > 0 && (
            <div className="mx-auto mb-2 flex w-full max-w-2xl flex-wrap gap-2">
              {pending.map((x) => (
                <div
                  key={x.key}
                  title={x.error ?? x.name}
                  className={`flex items-center gap-2 rounded-xl border px-2 py-1 text-xs ${
                    x.status === "error" ? "border-red-800 text-red-300" : "border-zinc-700 text-zinc-300"
                  }`}
                >
                  {x.preview ? (
                    <img src={x.preview} alt="" className="h-8 w-8 rounded object-cover" />
                  ) : (
                    <span>📄</span>
                  )}
                  <span className="max-w-[10rem] truncate">{x.name}</span>
                  {x.status === "uploading" && <span className="text-zinc-500">…</span>}
                  {x.status === "error" && <span className="max-w-[12rem] truncate">{x.error}</span>}
                  <button onClick={() => removePending(x.key)} className="text-zinc-500 hover:text-zinc-200">
                    ✕
                  </button>
                </div>
              ))}
            </div>
          )}
          <div className="mx-auto flex w-full max-w-2xl items-end gap-2">
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept={ACCEPT}
              className="hidden"
              onChange={(e) => {
                void addFiles(Array.from(e.target.files ?? []));
                e.target.value = "";
              }}
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              title="Attach images, text or code files (you can also paste or drop them)"
              className="rounded-2xl border border-zinc-700 px-3 py-3 text-sm text-zinc-300 hover:bg-zinc-900"
            >
              📎
            </button>
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              onPaste={onPaste}
              placeholder="Send a message…  (Enter to send, Shift+Enter for newline)"
              rows={1}
              className="flex-1 resize-none rounded-2xl border border-zinc-700 bg-zinc-900 px-4 py-3 text-sm text-zinc-100 outline-none focus:border-emerald-600 focus:ring-1 focus:ring-emerald-600"
            />
            {busy ? (
              <button
                onClick={stop}
                className="rounded-2xl border border-zinc-700 bg-zinc-800 px-4 py-3 text-sm font-medium text-zinc-100 hover:bg-zinc-700"
              >
                Stop
              </button>
            ) : (
              <button
                onClick={() => void send()}
                disabled={(!input.trim() && !pending.some((x) => x.status === "ready")) || pending.some((x) => x.status === "uploading") || !!loadingModel}
                className="rounded-2xl bg-emerald-600 px-4 py-3 text-sm font-semibold text-white hover:bg-emerald-500 disabled:opacity-40"
              >
                Send
              </button>
            )}
          </div>
        </div>
      </div>
      {modelsOpen && (
        <ModelsPanel
          activeModel={model}
          activeVision={vision}
          onClose={() => setModelsOpen(false)}
          onModelChanged={() => void refreshHealth()}
        />
      )}
    </div>
  );
}
