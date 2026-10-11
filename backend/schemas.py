"""Pydantic schemas for the LocalChat API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ConversationCreate(BaseModel):
    title: str = ""
    system_prompt: str = ""
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(-1, ge=-1)


class ConversationUpdate(BaseModel):
    title: str | None = None
    system_prompt: str | None = None
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    top_p: float | None = Field(None, ge=0.0, le=1.0)
    max_tokens: int | None = Field(None, ge=-1)


class ChatRequest(BaseModel):
    """Stream a reply into an existing conversation.

    The client sends only the new user message (``content``), never the full
    history — the database is the source of truth.  ``regenerate: true``
    deletes the last assistant message and re-answers from the remaining
    history without adding a new user message.
    """
    conversation_id: int
    content: str = ""
    regenerate: bool = False
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    top_p: float | None = Field(None, ge=0.0, le=1.0)
    max_tokens: int | None = Field(None, ge=-1)