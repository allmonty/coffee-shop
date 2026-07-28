"""A chat model that returns exactly what the test tells it to.

This is what makes the graph tests fast, deterministic and CI-safe: every
assertion about routing, tool wiring and the confirmation gates runs with no
Ollama present. The real model is exercised only by the scenario suite.

LangChain ships fake chat models, but none that emit tool calls on a script, so
this is ~40 lines of our own rather than a fight with theirs.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


def tool_call(name: str, **arguments) -> AIMessage:
    """An assistant turn that asks for one tool."""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": arguments, "id": f"call_{uuid.uuid4().hex[:8]}"}],
    )


def tool_calls(*calls: tuple[str, dict]) -> AIMessage:
    """Several tools asked for in ONE assistant turn.

    Real models do this constantly ("add it and charge me"), and it is the case
    that exposed the concurrent-tool-execution bug.
    """
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:8]}"}
            for name, args in calls
        ],
    )


def says(text: str) -> AIMessage:
    """An assistant turn that just talks — this is what ends a turn."""
    return AIMessage(content=text)


class FakeToolCallingModel(BaseChatModel):
    """Replays `script` one message per invocation.

    Running past the end of the script raises rather than looping forever, so a
    graph that fails to terminate fails the test instead of hanging it.
    """

    script: list[AIMessage]
    calls: list[list[BaseMessage]] = []

    def __init__(self, script: Sequence[AIMessage], **kwargs: Any):
        super().__init__(script=list(script), calls=[], **kwargs)

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling"

    def bind_tools(self, tools, **kwargs):  # noqa: ARG002
        # The graph binds tools; the script already decides what gets called.
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        if not self.script:
            raise AssertionError(
                "FakeToolCallingModel ran out of scripted replies — "
                "the graph looped more times than the test expected"
            )
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])
