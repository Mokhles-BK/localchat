"""Conversation CRUD routes for the LocalChat API."""

from __future__ import annotations

import json
import re

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

import storage
from schemas import ConversationCreate, ConversationUpdate, ForkRequest

router = APIRouter(prefix="/api/conversations", tags=["conversations"])
search_router = APIRouter(prefix="/api/search", tags=["search"])


@router.get("")
def list_conversations_endpoint():
    return {"conversations": storage.list_conversations()}


@router.post("", status_code=201)
def create_conversation_endpoint(body: ConversationCreate):
    conv = storage.create_conversation(
        title=body.title,
        system_prompt=body.system_prompt,
        temperature=body.temperature,
        top_p=body.top_p,
        max_tokens=body.max_tokens,
    )
    return conv


@router.get("/{conv_id}")
def get_conversation_endpoint(conv_id: int):
    conv = storage.get_conversation(conv_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return conv


@router.patch("/{conv_id}")
def update_conversation_endpoint(conv_id: int, body: ConversationUpdate):
    if storage.get_conversation(conv_id) is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    # Only pass fields the client actually set (non-None).
    updates = body.model_dump(exclude_none=True)
    conv = storage.update_conversation(conv_id, **updates)
    return conv


@router.delete("/{conv_id}", status_code=204)
def delete_conversation_endpoint(conv_id: int):
    if not storage.delete_conversation(conv_id):
        raise HTTPException(status_code=404, detail="conversation not found")


@router.get("/{conv_id}/messages")
def list_messages_endpoint(conv_id: int):
    if storage.get_conversation(conv_id) is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return {"messages": storage.list_messages(conv_id)}


@router.post("/{conv_id}/fork", status_code=201)
def fork_conversation_endpoint(conv_id: int, body: ForkRequest):
    """Branch a conversation: copy it up to (and including) one message."""
    new = storage.fork_conversation(conv_id, body.up_to_message_id)
    if new is None:
        raise HTTPException(status_code=404, detail="conversation or message not found")
    return new


def _safe_filename(title: str, conv_id: int) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", title).strip("._")[:50]
    return name or f"conversation-{conv_id}"


@router.get("/{conv_id}/export")
def export_conversation_endpoint(conv_id: int, format: str = "md"):
    conv = storage.get_conversation(conv_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    if format not in ("md", "json"):
        raise HTTPException(status_code=400, detail="format must be md or json")
    messages = storage.list_messages(conv_id)
    filename = f"{_safe_filename(conv['title'], conv_id)}.{format}"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}

    if format == "json":
        payload = {
            "conversation": conv,
            "messages": [{"role": m["role"], "content": m["content"], "meta": m["meta"]} for m in messages],
        }
        return Response(
            json.dumps(payload, ensure_ascii=False, indent=2),
            media_type="application/json",
            headers=headers,
        )

    lines = [f"# {conv['title'] or 'Conversation'}", ""]
    lines.append(f"*temperature {conv['temperature']} · top_p {conv['top_p']}*")
    lines.append("")
    if conv["system_prompt"]:
        lines.append("**System prompt**")
        lines.append("")
        lines.extend(f"> {ln}" for ln in conv["system_prompt"].splitlines() or [""])
        lines.append("")
    for m in messages:
        lines.append("## You" if m["role"] == "user" else "## Assistant")
        lines.append("")
        lines.append(m["content"])
        lines.append("")
    return Response("\n".join(lines), media_type="text/markdown; charset=utf-8", headers=headers)


@search_router.get("")
def search_endpoint(q: str = ""):
    """Search conversation titles and message text."""
    return {"results": storage.search(q)}


__all__ = ["router", "search_router"]