# LocalChat

An Ollama-style local chatbot: runs open LLMs (GGUF) on your own machine,
with a model manager, a polished streaming chat UI, conversation history,
and an OpenAI-compatible local API.

**Inference engine:** llama.cpp `llama-server` (prebuilt Windows CUDA release),
run as a managed subprocess. We do not write our own inference engine and we do
not use llama-cpp-python (Windows CUDA build issues).

**Backend:** Python 3.11 + FastAPI. Responsibilities: start/stop/swap
llama-server, model manager (list/download/delete GGUF files from Hugging Face
via `huggingface_hub`), conversation storage (SQLite), streaming chat via
Server-Sent Events, OpenAI-compatible endpoints (`/v1/chat/completions`,
`/v1/models`).

**Frontend:** Vite + React + TypeScript + Tailwind. Streaming token display,
markdown and code-block rendering, chat sidebar, model picker, settings panel.

**Storage:** `data/` folder (gitignored) holds models, SQLite DB, logs.

## Repo layout

```
localchat/
├── CLAUDE.md
├── README.md
├── .gitignore
├── scripts/        # setup and helper scripts (PowerShell)
├── backend/        # FastAPI app, tests
├── frontend/       # Vite React app
└── data/           # gitignored: models/, db, logs, bin/
```

## Quick start

```powershell
# One-time setup: checks GPU, downloads llama-server + a default model,
# creates the Python venv, installs backend + frontend deps.
./scripts/setup.ps1

# Start backend + frontend together.
./scripts/dev.ps1
```

Then open http://localhost:5173 in your browser.

## Hardware note

Target quantized models of 1B–4B parameters (Q4_K_M). Always offload as many
layers to the GPU as fit; never assume more than 4 GB VRAM.

## Roadmap

1. Foundation: setup script, backend runs llama-server, minimal chat endpoint
   with streaming, minimal UI that streams a reply.
2. Chat app: conversation history in SQLite, multiple chats, system prompts,
   markdown/code rendering, stop-generation button, generation settings.
3. Model manager: browse/download GGUF models from Hugging Face with progress,
   delete, switch models at runtime, VRAM-aware GPU layer suggestion.
4. API: OpenAI-compatible endpoints, API docs page, works with standard OpenAI
   client libraries.
5. Advanced features: document chat (RAG with local embeddings and a vector
   store), file uploads, optional web search tool, conversation export/import.
6. Polish: error handling, logging, packaging into a one-command start.