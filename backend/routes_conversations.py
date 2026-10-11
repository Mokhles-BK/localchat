"""Conversation CRUD routes for the LocalChat API."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

import storage
from schemas import ConversationCreate, ConversationUpdate

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


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


__all__ = ["router"]