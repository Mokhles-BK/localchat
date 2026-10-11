import { useRef, useState, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/github-dark.css";
import type { ChatMessage } from "../lib/api";

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
}

export default function Message({ message, isLast, streaming, onRegenerate }: Props) {
  const isAssistant = message.role === "assistant";
  const waiting = isAssistant && isLast && streaming && !message.content;

  return (
    <div className="flex min-w-0 flex-col gap-1">
      <span className="px-1 text-xs font-medium uppercase tracking-wide text-zinc-500">
        {message.role}
      </span>
      <div className="min-w-0 max-w-full overflow-hidden rounded-2xl bg-zinc-900 px-4 py-3 text-sm leading-relaxed text-zinc-100">
        {waiting ? (
          <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-zinc-500" />
        ) : (
          <div className="markdown">
            <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]} components={{ pre: CodeBlock }}>
              {message.content}
            </ReactMarkdown>
          </div>
        )}
      </div>
      {isAssistant && isLast && !streaming && message.content && (
        <button
          onClick={onRegenerate}
          className="self-start px-1 text-xs text-zinc-500 hover:text-zinc-300"
        >
          ↻ Regenerate
        </button>
      )}
    </div>
  );
}
