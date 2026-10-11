import { useState } from "react";
import type { Conversation } from "../lib/api";

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
      <div className="scroll-thin flex-1 overflow-y-auto px-2 pb-3">
        {conversations.length === 0 && (
          <p className="px-3 py-4 text-xs text-zinc-600">No conversations yet.</p>
        )}
        {conversations.map((c) => {
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
