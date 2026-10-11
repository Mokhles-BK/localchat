import type { ConversationSettings } from "../lib/api";

interface Props {
  settings: ConversationSettings;
  onChange: (patch: Partial<ConversationSettings>) => void;
}

export default function SettingsPanel({ settings, onChange }: Props) {
  return (
    <div className="border-b border-zinc-800 bg-zinc-950 px-5 py-4">
      <div className="mx-auto grid max-w-2xl gap-4">
        <label className="grid gap-1">
          <span className="text-xs font-medium uppercase tracking-wide text-zinc-500">System prompt</span>
          <textarea
            value={settings.system_prompt}
            onChange={(e) => onChange({ system_prompt: e.target.value })}
            rows={3}
            placeholder="e.g. You are a concise assistant. Answer in the user's language."
            className="resize-y rounded-xl border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-emerald-600"
          />
        </label>

        <div className="grid gap-4 sm:grid-cols-3">
          <label className="grid gap-1">
            <span className="flex justify-between text-xs text-zinc-500">
              <span>Temperature</span>
              <span className="text-zinc-300">{settings.temperature.toFixed(2)}</span>
            </span>
            <input
              type="range"
              min={0}
              max={2}
              step={0.05}
              value={settings.temperature}
              onChange={(e) => onChange({ temperature: Number(e.target.value) })}
              className="accent-emerald-500"
            />
          </label>

          <label className="grid gap-1">
            <span className="flex justify-between text-xs text-zinc-500">
              <span>Top P</span>
              <span className="text-zinc-300">{settings.top_p.toFixed(2)}</span>
            </span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={settings.top_p}
              onChange={(e) => onChange({ top_p: Number(e.target.value) })}
              className="accent-emerald-500"
            />
          </label>

          <label className="grid gap-1">
            <span className="text-xs text-zinc-500">Max tokens (-1 = auto)</span>
            <input
              type="number"
              min={-1}
              step={64}
              value={settings.max_tokens}
              onChange={(e) => {
                const n = Math.round(Number(e.target.value));
                if (Number.isFinite(n) && n >= -1) onChange({ max_tokens: n });
              }}
              className="rounded-lg border border-zinc-700 bg-zinc-900 px-2 py-1 text-sm text-zinc-100 outline-none focus:border-emerald-600"
            />
          </label>
        </div>
      </div>
    </div>
  );
}
