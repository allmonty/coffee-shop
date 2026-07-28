"""Pydantic request/response types for the REST surface (spec §7.1)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EnterRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)


class ChatRequest(BaseModel):
    """`message` for typed input, `event` for non-typed triggers.

    `on_enter` makes the barista greet first; `go_home` is the button, which is
    an unambiguous exit that never depends on the model interpreting "bye".
    """

    visit_id: str
    message: str | None = None
    event: str | None = None
