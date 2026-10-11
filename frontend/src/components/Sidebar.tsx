import { useEffect, useState } from "react";
import { api, type Conversation, type SearchHit } from "../lib/api";

interface Props {
  conversations: Conversation[];
  activeId: number | null;
  onSelect: (id: number) => void;
  onNew: () => void;
  onRename: (id: number, title: string) => void;
  onDelete: (id: number) => void;
}

export default function Sidebar({ conversations, activeId, onSelect, onNew, onRename, onDelete }: Props) {
  const [editingId, setEditingId] = useState<number | null>(null);
  const [draft, setDraft] = useState("");
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<SearchHit[] | null>(null);

  // Debounced search across titles and message text.
  useEffect(() => {
    const q = query.trim();
    if (!q) {
      setHits(null);
      return;
    }
    let stale = false;
    const t = window.setTimeout(() => {
      api
        .search(q)
        .then((r) => {
          if (!stale) setHits(r);
        })
        .catch(() => {
          if (!stale) setHits([]);
        });
    }, 250);
    return () => {
      stale = true;
      window.clearTimeout(t);
    };
  }, [query]);

  function startRename(c: Conversation) {
    setEditingId(c.id);
    setDraft(c.title);
  }

  function commitRename() {
    if (editingId !== null) {
      const title = draft.trim();
      const current = conversations.find((c) => c.id === editingId);
      if (title && current && title !== current.title) onRename(editingId, title);
    }
    setEditingId(null);
  }

  return (
    <aside className="flex h-full w-64 shrink-0 flex-col border-r border-zinc-800 bg-zinc-950">
      <div className="p-3">
        <button
          onClick={onNew}
          className="w-full rounded-xl border border-zinc-700 px-3 py-2 text-sm font-medium text-zinc-100 hover:bg-zinc-900"
        >
          + New chat
        </button>
      </div>
      <div className="px-3 pb-2">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Escape" && setQuery("")}
          placeholder="Search chats…"
          className="w-full rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-1.5 text-sm text-zinc-100 outline-none focus:border-emerald-600"
        />
      </div>
      <div className="scroll-thin flex-1 overflow-y-auto px-2 pb-3">
        {hits !== null && (
          <div>
            {hits.length === 0 && <p className="px-3 py-4 text-xs text-zinc-600">No matches.</p>}
            {hits.map((h, i) => (
              <button
                key={`${h.conversation_id}-${h.message_id ?? "t"}-${i}`}
                onClick={() => {
                  onSelect(h.conversation_id);
                  setQuery("");
                }}
                className="mb-1 block w-full rounded-lg px-3 py-2 text-left hover:bg-zinc-900"
              >
                <div className="truncate text-xs font-medium text-zinc-300">{h.title || "New chat"}</div>
                <div className="mt-0.5 line-clamp-2 text-xs text-zinc-500">
                  {h.role ? `${h.role === "user" ? "You" : "AI"}: ` : "Title: "}
                  {h.snippet}
                </div>
              </button>
            ))}
          </div>
        )}
        {hits === null && conversations.length === 0 && (
          <p className="px-3 py-4 text-xs text-zinc-600">No conversations yet.</p>
        )}
        {hits === null && conversations.map((c) => {
          const active = c.id === activeId;
          return (
            <div
              key={c.id}
              className={`group mb-1 flex items-center gap-1 rounded-lg px-3 py-2 text-sm ${
                active ? "bg-zinc-800 text-zinc-100" : "text-zinc-400 hover:bg-zinc-900"
              }`}
            >
              {editingId === c.id ? (
                <input
                  autoFocus
                  value={draft}
                  onChange={(e) => setDraft(e.target.value)}
                  onBlur={commitRename}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") commitRename();
                    if (e.key === "Escape") setEditingId(null);
                  }}
                  className="min-w-0 flex-1 rounded bg-zinc-900 px-1 text-sm text-zinc-100 outline-none ring-1 ring-emerald-600"
                />
              ) : (
                <button
                  onClick={() => onSelect(c.id)}
                  onDoubleClick={() => startRename(c)}
                  title="Double-click to rename"
                  className="min-w-0 flex-1 truncate text-left"
                >
                  {c.title || "New chat"}
                </button>
              )}
              {editingId !== c.id && (
                <div className="flex shrink-0 gap-1 opacity-0 group-hover:opacity-100">
                  <button
                    onClick={() => startRename(c)}
                    title="Rename"
                    className="rounded px-1 text-xs text-zinc-500 hover:text-zinc-200"
                  >
                    ✎
                  </button>
                  <button
                    onClick={() => onDelete(c.id)}
                    title="Delete"
                    className="rounded px-1 text-xs text-zinc-500 hover:text-red-400"
                  >
                    ✕
                  </button>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </aside>
  );
}
