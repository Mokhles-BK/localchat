import { useRef, useState, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/github-dark.css";
import { attachmentUrl, type ChatMessage } from "../lib/api";

/** <pre> wrapper that adds a Copy button to every code block. */
function CodeBlock({ children }: { children?: ReactNode }) {
  const ref = useRef<HTMLPreElement>(null);
  const [copied, setCopied] = useState(false);

  async function copy() {
    const text = ref.current?.innerText ?? "";
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      // clipboard blocked — nothing sensible to do
    }
  }

  return (
    <div className="group relative">
      <button
        onClick={copy}
        className="absolute right-2 top-2 z-10 rounded-md border border-zinc-700 bg-zinc-800 px-2 py-0.5 text-xs text-zinc-300 opacity-0 transition-opacity hover:bg-zinc-700 group-hover:opacity-100"
      >
        {copied ? "Copied" : "Copy"}
      </button>
      <pre ref={ref}>{children}</pre>
    </div>
  );
}

interface Props {
  message: ChatMessage;
  isLast: boolean;
  streaming: boolean;
  onRegenerate: () => void;
  /** Present only for user messages that can be edited right now. */
  onEdit?: (text: string) => void;
  /** Present only for messages that can start a branch right now. */
  onFork?: () => void;
}

const actionCls = "text-xs text-zinc-500 hover:text-zinc-300";

export default function Message({ message, isLast, streaming, onRegenerate, onEdit, onFork }: Props) {
  const isAssistant = message.role === "assistant";
  const waiting = isAssistant && isLast && streaming && !message.content;
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const [copied, setCopied] = useState(false);
  const meta = message.meta;

  function startEdit() {
    setDraft(message.content);
    setEditing(true);
  }

  function saveEdit() {
    const text = draft.trim();
    if (!text || !onEdit) return;
    setEditing(false);
    if (text !== message.content) onEdit(text);
  }

  async function copy() {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      // clipboard blocked
    }
  }

  return (
    <div className="group/msg flex min-w-0 flex-col gap-1">
      <span className="px-1 text-xs font-medium uppercase tracking-wide text-zinc-500">{message.role}</span>

      {editing ? (
        <div className="grid gap-2">
          <textarea
            autoFocus
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) saveEdit();
              if (e.key === "Escape") setEditing(false);
            }}
            rows={Math.min(10, Math.max(3, draft.split("\n").length))}
            className="w-full resize-y rounded-2xl border border-emerald-700 bg-zinc-900 px-4 py-3 text-sm text-zinc-100 outline-none"
          />
          <div className="flex items-center gap-3 px-1">
            <button
              onClick={saveEdit}
              disabled={!draft.trim()}
              className="rounded-lg bg-emerald-600 px-3 py-1 text-xs font-semibold text-white hover:bg-emerald-500 disabled:opacity-40"
            >
              Save &amp; resend
            </button>
            <button onClick={() => setEditing(false)} className={actionCls}>
              Cancel
            </button>
            <span className="text-xs text-zinc-600">Replies after this message will be replaced. Ctrl+Enter saves.</span>
          </div>
        </div>
      ) : (
        <div className="min-w-0 max-w-full overflow-hidden rounded-2xl bg-zinc-900 px-4 py-3 text-sm leading-relaxed text-zinc-100">
          {message.attachments && message.attachments.length > 0 && (
            <div className={`flex flex-wrap gap-2 ${message.content ? "mb-3" : ""}`}>
              {message.attachments.map((a) =>
                a.kind === "image" ? (
                  <a key={a.id} href={attachmentUrl(a.id)} target="_blank" rel="noreferrer" title={a.name}>
                    <img
                      src={attachmentUrl(a.id)}
                      alt={a.name}
                      className="max-h-48 max-w-full rounded-lg border border-zinc-700 object-contain"
                    />
                  </a>
                ) : (
                  <a
                    key={a.id}
                    href={attachmentUrl(a.id)}
                    target="_blank"
                    rel="noreferrer"
                    className="flex items-center gap-2 rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-1.5 text-xs text-zinc-300 hover:bg-zinc-800"
                  >
                    <span>📄</span>
                    <span className="max-w-[12rem] truncate">{a.name}</span>
                  </a>
                ),
              )}
            </div>
          )}
          {waiting ? (
            <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-zinc-500" />
          ) : message.content ? (
            <div className="markdown">
              <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]} components={{ pre: CodeBlock }}>
                {message.content}
              </ReactMarkdown>
            </div>
          ) : null}
        </div>
      )}

      {!editing && !streaming && message.content && (
        <div className="flex flex-wrap items-center gap-3 px-1">
          {isAssistant && meta && (
            <span className="text-xs text-zinc-600">
              {meta.exact ? "" : "≈ "}
              {meta.tokens} tokens{meta.tps ? ` · ${meta.tps} tok/s` : ""}
            </span>
          )}
          <span className="flex gap-3 opacity-0 transition-opacity group-hover/msg:opacity-100 focus-within:opacity-100">
            <button onClick={() => void copy()} className={actionCls}>
              {copied ? "Copied" : "Copy"}
            </button>
            {onEdit && (
              <button onClick={startEdit} className={actionCls}>
                ✎ Edit
              </button>
            )}
            {onFork && (
              <button onClick={onFork} className={actionCls} title="Continue from here in a new chat">
                ⑂ Branch
              </button>
            )}
          </span>
          {isAssistant && isLast && (
            <button onClick={onRegenerate} className={actionCls}>
              ↻ Regenerate
            </button>
          )}
        </div>
      )}
    </div>
  );
}
