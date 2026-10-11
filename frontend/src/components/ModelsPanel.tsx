import { useEffect, useRef, useState } from "react";
import {
  api,
  formatBytes,
  type DownloadJob,
  type Fit,
  type LocalCandidate,
  type ModelInfo,
  type RepoFile,
  type RepoResult,
} from "../lib/api";

interface Props {
  activeModel: string | null;
  onClose: () => void;
  onModelChanged: () => void;
}

function FitBadge({ fit }: { fit?: Fit }) {
  if (!fit || fit === "unknown") return null;
  const map: Record<Exclude<Fit, "unknown">, [string, string]> = {
    gpu: ["fits GPU", "bg-emerald-950 text-emerald-400"],
    partial: ["partial GPU", "bg-amber-950 text-amber-400"],
    too_big: ["too big, slow", "bg-red-950 text-red-400"],
  };
  const [label, cls] = map[fit];
  return <span className={`rounded px-1.5 py-0.5 text-[10px] ${cls}`}>{label}</span>;
}

const isActive = (j: DownloadJob) => j.status === "queued" || j.status === "downloading";
const isArmOnly = (name: string) => /Q4_0_\d_\d/i.test(name);

export default function ModelsPanel({ activeModel, onClose, onModelChanged }: Props) {
  const [local, setLocal] = useState<ModelInfo[]>([]);
  const [jobs, setJobs] = useState<DownloadJob[]>([]);
  const [dismissed, setDismissed] = useState<Set<string>>(new Set());
  const [q, setQ] = useState("");
  const [results, setResults] = useState<RepoResult[] | null>(null);
  const [repo, setRepo] = useState<string | null>(null);
  const [files, setFiles] = useState<RepoFile[] | null>(null);
  const [searching, setSearching] = useState(false);
  const [loadingFiles, setLoadingFiles] = useState(false);
  const [switching, setSwitching] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [candidates, setCandidates] = useState<LocalCandidate[] | null>(null);
  const [scanning, setScanning] = useState(false);
  const [folder, setFolder] = useState("");
  const prevStatus = useRef<Record<string, string>>({});

  const fail = (e: unknown) => setError((e as Error).message);

  async function refreshLocal() {
    try {
      setLocal(await api.listModels());
    } catch (e) {
      fail(e);
    }
  }

  useEffect(() => {
    void refreshLocal();
  }, []);

  // Poll download progress; reload the installed list when one finishes.
  useEffect(() => {
    let stopped = false;
    async function tick() {
      try {
        const list = await api.downloads();
        if (stopped) return;
        setJobs(list);
        let finished = false;
        for (const j of list) {
          if (prevStatus.current[j.id] && prevStatus.current[j.id] !== "done" && j.status === "done") finished = true;
          prevStatus.current[j.id] = j.status;
        }
        if (finished) {
          void refreshLocal();
          if (repo) void openRepo(repo);
        }
      } catch {
        // backend busy or restarting — try again next tick
      }
    }
    void tick();
    const id = window.setInterval(tick, 1500);
    return () => {
      stopped = true;
      window.clearInterval(id);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [repo]);

  async function search() {
    if (!q.trim()) return;
    setSearching(true);
    setError(null);
    setRepo(null);
    setFiles(null);
    try {
      setResults(await api.searchRepos(q.trim()));
    } catch (e) {
      fail(e);
    } finally {
      setSearching(false);
    }
  }

  async function openRepo(id: string) {
    setRepo(id);
    setLoadingFiles(true);
    setError(null);
    try {
      setFiles(await api.repoFiles(id));
    } catch (e) {
      fail(e);
      setFiles(null);
    } finally {
      setLoadingFiles(false);
    }
  }

  async function scan(useFolder: boolean) {
    setScanning(true);
    setError(null);
    try {
      setCandidates(await api.scan(useFolder ? folder.trim() : undefined));
    } catch (e) {
      fail(e);
    } finally {
      setScanning(false);
    }
  }

  async function importModel(c: LocalCandidate) {
    setError(null);
    try {
      await api.importModel(c.path, c.name);
      setJobs(await api.downloads());
    } catch (e) {
      fail(e);
    }
  }

  async function download(f: RepoFile) {
    if (!repo) return;
    setError(null);
    try {
      await api.startDownload(repo, f.filename);
      setJobs(await api.downloads());
    } catch (e) {
      fail(e);
    }
  }

  async function cancel(id: string) {
    try {
      await api.cancelDownload(id);
    } catch (e) {
      fail(e);
    }
  }

  async function use(name: string) {
    setSwitching(name);
    setError(null);
    try {
      await api.activateModel(name);
      onModelChanged();
    } catch (e) {
      fail(e);
      onModelChanged();
    } finally {
      setSwitching(null);
      void refreshLocal();
    }
  }

  async function remove(name: string) {
    if (!window.confirm(`Delete ${name} from disk?`)) return;
    try {
      await api.deleteModel(name);
      await refreshLocal();
    } catch (e) {
      fail(e);
    }
  }

  const installedNames = new Set(local.map((m) => m.name));
  const sourceLabel = { ollama: "Ollama", lmstudio: "LM Studio", folder: "Folder" } as const;

  const visibleJobs = jobs.filter((j) => isActive(j) || (j.status === "error" && !dismissed.has(j.id)));

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-black/70 p-4 sm:p-10">
      <div className="w-full max-w-2xl rounded-2xl border border-zinc-800 bg-zinc-950 shadow-2xl">
        <div className="flex items-center justify-between border-b border-zinc-800 px-5 py-4">
          <h2 className="text-sm font-semibold text-zinc-100">Models</h2>
          <button onClick={onClose} className="rounded px-2 py-1 text-zinc-400 hover:bg-zinc-900 hover:text-zinc-100">
            ✕
          </button>
        </div>

        <div className="grid gap-6 px-5 py-4">
          {error && (
            <div className="rounded-xl border border-red-800/60 bg-red-950/40 px-4 py-3 text-sm text-red-300">{error}</div>
          )}

          {switching && (
            <div className="rounded-xl border border-emerald-800/60 bg-emerald-950/30 px-4 py-3 text-sm text-emerald-300">
              Loading {switching}… this takes 10 to 60 seconds. Chat is paused until it is ready.
            </div>
          )}

          {/* Installed */}
          <section>
            <h3 className="mb-2 text-xs font-medium uppercase tracking-wide text-zinc-500">Installed</h3>
            {local.length === 0 && <p className="text-sm text-zinc-600">No models yet. Download one below.</p>}
            <div className="grid gap-2">
              {local.map((m) => {
                const active = m.name === activeModel;
                return (
                  <div key={m.name} className="flex items-center gap-3 rounded-xl bg-zinc-900 px-3 py-2">
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-sm text-zinc-100">{m.name}</div>
                      <div className="mt-0.5 flex items-center gap-2 text-xs text-zinc-500">
                        <span>{formatBytes(m.size_bytes)}</span>
                        {m.quant && <span>{m.quant}</span>}
                        <FitBadge fit={m.fit} />
                      </div>
                    </div>
                    {active ? (
                      <span className="rounded-lg bg-emerald-900/50 px-2 py-1 text-xs text-emerald-300">Active</span>
                    ) : (
                      <>
                        <button
                          onClick={() => void use(m.name)}
                          disabled={switching !== null}
                          className="rounded-lg bg-emerald-600 px-3 py-1 text-xs font-semibold text-white hover:bg-emerald-500 disabled:opacity-40"
                        >
                          Use
                        </button>
                        <button
                          onClick={() => void remove(m.name)}
                          disabled={switching !== null}
                          title="Delete from disk"
                          className="rounded px-1 text-xs text-zinc-500 hover:text-red-400 disabled:opacity-40"
                        >
                          ✕
                        </button>
                      </>
                    )}
                  </div>
                );
              })}
            </div>
          </section>

          {/* Active downloads */}
          {visibleJobs.length > 0 && (
            <section>
              <h3 className="mb-2 text-xs font-medium uppercase tracking-wide text-zinc-500">Downloads</h3>
              <div className="grid gap-2">
                {visibleJobs.map((j) => {
                  const pct = j.total > 0 ? Math.min(100, (j.downloaded / j.total) * 100) : 0;
                  return (
                    <div key={j.id} className="rounded-xl bg-zinc-900 px-3 py-2">
                      <div className="flex items-center justify-between gap-2 text-sm">
                        <span className="min-w-0 truncate text-zinc-200">{j.filename}</span>
                        {isActive(j) ? (
                          <button onClick={() => void cancel(j.id)} className="text-xs text-zinc-500 hover:text-red-400">
                            Cancel
                          </button>
                        ) : (
                          <button
                            onClick={() => setDismissed((d) => new Set(d).add(j.id))}
                            className="text-xs text-zinc-500 hover:text-zinc-300"
                          >
                            Dismiss
                          </button>
                        )}
                      </div>
                      {isActive(j) ? (
                        <>
                          <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-zinc-800">
                            <div className="h-full rounded-full bg-emerald-500 transition-all" style={{ width: `${pct}%` }} />
                          </div>
                          <div className="mt-1 text-xs text-zinc-500">
                            {j.total > 0 ? `${formatBytes(j.downloaded)} / ${formatBytes(j.total)} (${pct.toFixed(0)}%)` : "starting…"}
                          </div>
                        </>
                      ) : (
                        <div className="mt-1 text-xs text-red-400">{j.error}</div>
                      )}
                    </div>
                  );
                })}
              </div>
            </section>
          )}

          {/* Already on this PC */}
          <section>
            <h3 className="mb-2 text-xs font-medium uppercase tracking-wide text-zinc-500">Found on this PC</h3>
            <p className="mb-2 text-xs text-zinc-600">
              Import models you already have (Ollama, LM Studio, or any folder). On the same drive this is instant and
              uses no extra disk space.
            </p>
            <div className="flex gap-2">
              <button
                onClick={() => void scan(false)}
                disabled={scanning}
                className="shrink-0 rounded-xl bg-zinc-800 px-4 py-2 text-sm text-zinc-100 hover:bg-zinc-700 disabled:opacity-40"
              >
                {scanning ? "Scanning…" : "Scan Ollama / LM Studio"}
              </button>
              <input
                value={folder}
                onChange={(e) => setFolder(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && folder.trim() && void scan(true)}
                placeholder="or a folder, e.g. C:\Users\you\Downloads"
                className="min-w-0 flex-1 rounded-xl border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-emerald-600"
              />
              <button
                onClick={() => void scan(true)}
                disabled={scanning || !folder.trim()}
                className="shrink-0 rounded-xl bg-zinc-800 px-3 py-2 text-sm text-zinc-100 hover:bg-zinc-700 disabled:opacity-40"
              >
                Scan folder
              </button>
            </div>
            {candidates && candidates.length === 0 && (
              <p className="mt-3 text-sm text-zinc-600">No GGUF models found there.</p>
            )}
            {candidates && candidates.length > 0 && (
              <div className="mt-3 grid gap-1">
                {candidates.map((c) => {
                  const have = c.installed || installedNames.has(c.name);
                  const importing = jobs.some((j) => j.filename === c.name && isActive(j));
                  return (
                    <div key={c.path} className="flex items-center gap-3 rounded-lg px-3 py-2 hover:bg-zinc-900">
                      <div className="min-w-0 flex-1">
                        <div className="truncate text-sm text-zinc-200">{c.name}</div>
                        <div className="mt-0.5 flex flex-wrap items-center gap-2 text-xs text-zinc-500">
                          <span>{sourceLabel[c.source]}</span>
                          <span>{formatBytes(c.size_bytes)}</span>
                          {c.quant && <span>{c.quant}</span>}
                          <FitBadge fit={c.fit} />
                        </div>
                      </div>
                      {have ? (
                        <span className="text-xs text-zinc-500">Imported</span>
                      ) : (
                        <button
                          onClick={() => void importModel(c)}
                          disabled={importing}
                          className="rounded-lg bg-zinc-800 px-3 py-1 text-xs text-zinc-100 hover:bg-zinc-700 disabled:opacity-40"
                        >
                          {importing ? "Importing…" : "Import"}
                        </button>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </section>

          {/* Hugging Face */}
          <section>
            <h3 className="mb-2 text-xs font-medium uppercase tracking-wide text-zinc-500">Download from Hugging Face</h3>
            <div className="flex gap-2">
              <input
                value={q}
                onChange={(e) => setQ(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && void search()}
                placeholder="Search GGUF models, e.g. qwen2.5 3b instruct"
                className="flex-1 rounded-xl border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-emerald-600"
              />
              <button
                onClick={() => void search()}
                disabled={searching || !q.trim()}
                className="rounded-xl bg-zinc-800 px-4 py-2 text-sm text-zinc-100 hover:bg-zinc-700 disabled:opacity-40"
              >
                {searching ? "…" : "Search"}
              </button>
            </div>

            {results && !repo && (
              <div className="mt-3 grid gap-1">
                {results.length === 0 && <p className="text-sm text-zinc-600">No results.</p>}
                {results.map((r) => (
                  <button
                    key={r.repo_id}
                    onClick={() => void openRepo(r.repo_id)}
                    className="flex items-center justify-between rounded-lg px-3 py-2 text-left text-sm text-zinc-200 hover:bg-zinc-900"
                  >
                    <span className="min-w-0 truncate">{r.repo_id}</span>
                    <span className="shrink-0 text-xs text-zinc-500">
                      ↓ {r.downloads.toLocaleString()} · ♥ {r.likes}
                    </span>
                  </button>
                ))}
              </div>
            )}

            {repo && (
              <div className="mt-3">
                <button
                  onClick={() => {
                    setRepo(null);
                    setFiles(null);
                  }}
                  className="mb-2 text-xs text-zinc-500 hover:text-zinc-300"
                >
                  ← back to results
                </button>
                <div className="mb-2 truncate text-sm font-medium text-zinc-200">{repo}</div>
                {loadingFiles && <p className="text-sm text-zinc-600">Loading files…</p>}
                {files && files.length === 0 && <p className="text-sm text-zinc-600">No GGUF files in this repo.</p>}
                <div className="grid gap-1">
                  {files?.map((f) => {
                    const downloading = jobs.some((j) => j.filename === f.filename && isActive(j));
                    const arm = isArmOnly(f.filename);
                    return (
                      <div key={f.filename} className="flex items-center gap-3 rounded-lg px-3 py-2 hover:bg-zinc-900">
                        <div className="min-w-0 flex-1">
                          <div className="truncate text-sm text-zinc-200">{f.filename}</div>
                          <div className="mt-0.5 flex flex-wrap items-center gap-2 text-xs text-zinc-500">
                            <span>{formatBytes(f.size_bytes)}</span>
                            {f.quant && <span>{f.quant}</span>}
                            <FitBadge fit={f.fit} />
                            {arm && <span className="text-amber-500">ARM-only format</span>}
                            {f.note && <span>{f.note}</span>}
                          </div>
                        </div>
                        {f.installed ? (
                          <span className="text-xs text-zinc-500">Installed</span>
                        ) : (
                          <button
                            onClick={() => void download(f)}
                            disabled={!f.downloadable || downloading}
                            className="rounded-lg bg-zinc-800 px-3 py-1 text-xs text-zinc-100 hover:bg-zinc-700 disabled:opacity-40"
                          >
                            {downloading ? "Downloading…" : "Download"}
                          </button>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}
