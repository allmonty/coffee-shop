"""The chat model, pointed at Ollama's OpenAI-compatible API (spec §6.1).

Going through the OpenAI-compatible endpoint rather than an Ollama-specific
client is deliberate: when the barista starts behaving strangely, you can point
`OLLAMA_BASE_URL` at a hosted model and find out in one run whether the problem
is the graph or the model. That single trick saves more time than anything else
in this project.

`temperature=0` because this agent's job is to follow rules and call tools
correctly, not to be interesting. Personality comes from the prompt.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from settings import settings


def build_summary_llm(**overrides) -> ChatOpenAI:
    """The model for the visit-summarization pass (spec §6.5.1).

    Falls back to the conversation model when `OLLAMA_SUMMARY_MODEL` is unset,
    so the default behaviour is exactly what it was before this existed.
    """
    if settings.ollama_summary_model:
        overrides.setdefault("model", settings.ollama_summary_model)
    return build_llm(**overrides)


def build_llm(**overrides) -> ChatOpenAI:
    kwargs = {
        "model": settings.ollama_model,
        "base_url": settings.ollama_base_url,
        # Ollama ignores the key but the OpenAI client requires one.
        "api_key": "ollama",
        "temperature": 0,
        "timeout": 120,
    }
    kwargs.update(overrides)
    return ChatOpenAI(**kwargs)
