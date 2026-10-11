import { useEffect, useRef, useState } from "react";
import {
  api,
  type ChatMessage,
  type Conversation,
  type ConversationPatch,
  type ConversationSettings,
} from "./lib/api";
import Sidebar from "./components/Sidebar";
import SettingsPanel from "./components/SettingsPanel";
import Message from "./components/Message";

const ACTIVE_KEY = "localchat.activeId";
const DEFAULT_SETTINGS: ConversationSettings = {
  system_prompt: "",
  temperature: 0.7,
  top_p: 0.9,
  max_tokens: -1,
};

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export default function App() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveIdState] = useState<number | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState<ConversationSettings>(DEFAULT_SETTINGS);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [model, setModel] = useState<string | null>(null);
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
    api
      .health()
      .then((h) => setModel(h.model))
      .catch((e) => setError((e as Error).message));
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
  async function runStream(convId: number, body: { content?: string; regenerate?: boolean }) {
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

  async function send() {
    const text = input.trim();
    if (!text || busy) return;
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
    setMessages((m) => [...m, { role: "user", content: text }, { role: "assistant", content: "" }]);
    await runStream(convId, { content: text });
  }

  async function regenerate() {
    const convId = activeIdRef.current;
    if (convId === null || busy) return;
    setError(null);
    setMessages((m) => [...m.slice(0, -1), { role: "assistant", content: "" }]);
    await runStream(convId, { regenerate: true });
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
            <span className="text-xs text-zinc-500">{model ?? "loading model…"}</span>
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
              />
            ))}
            {error && (
              <div className="rounded-xl border border-red-800/60 bg-red-950/40 px-4 py-3 text-sm text-red-300">
                {error}
              </div>
            )}
          </div>
        </div>

        <div className="border-t border-zinc-800 px-4 py-4">
          <div className="mx-auto flex w-full max-w-2xl items-end gap-2">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
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
                disabled={!input.trim()}
                className="rounded-2xl bg-emerald-600 px-4 py-3 text-sm font-semibold text-white hover:bg-emerald-500 disabled:opacity-40"
              >
                Send
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
