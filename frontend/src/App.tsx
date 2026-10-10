import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { api, type ChatMessage } from "./lib/api";

type Msg = ChatMessage;

export default function App() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [model, setModel] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const listRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Auto-scroll to the bottom as tokens stream in.
  useEffect(() => {
    const el = listRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, busy]);

  // Fetch backend health / model name on mount.
  useEffect(() => {
    api
      .health()
      .then((h) => setModel(h.model))
      .catch((e) => setError((e as Error).message));
  }, []);

  async function send() {
    const text = input.trim();
    if (!text || busy) return;

    setError(null);
    setMessages((m) => [...m, { role: "user", content: text }]);
    setInput("");
    setBusy(true);

    const controller = new AbortController();
    abortRef.current = controller;

    // Optimistically add an assistant bubble that we'll stream into.
    setMessages((m) => [...m, { role: "assistant", content: "" }]);

    try {
      await api.streamChat(
        { messages: [...messages, { role: "user", content: text }] },
        (token) => {
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
      setError((e as Error).message);
      // Roll back the empty assistant bubble on hard failure.
      setMessages((m) => m.filter((x) => !(x.role === "assistant" && x.content === "")));
    } finally {
      setBusy(false);
      abortRef.current = null;
      inputRef.current?.focus();
    }
  }

  function stop() {
    abortRef.current?.abort();
    setBusy(false);
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  return (
    <div className="flex h-screen flex-col bg-zinc-950">
      {/* Header */}
      <header className="flex items-center justify-between border-b border-zinc-800 px-5 py-3">
        <div className="flex items-center gap-2">
          <div className="h-2.5 w-2.5 rounded-full bg-emerald-500" />
          <h1 className="text-sm font-semibold text-zinc-100">LocalChat</h1>
        </div>
        <div className="text-xs text-zinc-500">
          {model ? model : "loading model…"}
        </div>
      </header>

      {/* Messages */}
      <div ref={listRef} className="scroll-thin flex-1 min-w-0 overflow-y-auto px-4 py-6">
        <div className="mx-auto flex w-full max-w-2xl min-w-0 flex-col gap-5">
          {messages.length === 0 && !busy && (
            <div className="py-10 text-center text-zinc-600">
              <p className="text-base text-zinc-400">LocalChat is ready.</p>
              <p className="mt-1 text-sm">Send a message to stream a reply from the local model.</p>
            </div>
          )}
          {messages.map((m, i) => (
            <div key={i} className="flex min-w-0 flex-col gap-1">
              <span className="px-1 text-xs font-medium uppercase tracking-wide text-zinc-500">
                {m.role}
              </span>
              <div className="min-w-0 max-w-full overflow-hidden rounded-2xl bg-zinc-900 px-4 py-3 text-sm leading-relaxed text-zinc-100">
                {m.content ? (
                  <ReactMarkdown className="markdown" remarkPlugins={[remarkGfm]}>
                    {m.content}
                  </ReactMarkdown>
                ) : (
                  <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-zinc-600" />
                )}
              </div>
            </div>
          ))}
          {error && (
            <div className="rounded-xl border border-red-800/60 bg-red-950/40 px-4 py-3 text-sm text-red-300">
              {error}
            </div>
          )}
        </div>
      </div>

      {/* Composer */}
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
            style={{ maxHeight: "12rem" }}
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
              onClick={send}
              disabled={!input.trim()}
              className="rounded-2xl bg-emerald-600 px-4 py-3 text-sm font-semibold text-white disabled:opacity-40 hover:bg-emerald-500"
            >
              Send
            </button>
          )}
        </div>
      </div>
    </div>
  );
}