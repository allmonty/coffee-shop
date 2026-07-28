"""Does the local model actually emit a well-formed tool call?

Run this BEFORE writing any graph code, and again whenever you change
`OLLAMA_MODEL`:

    uv run python scripts/check_tool_calling.py

No LangGraph, no LangChain, no application code — a raw OpenAI-compatible
request with one tool schema. That is the whole point. The most common way to
lose a day on an agent project is debugging a graph when the real problem is a
model that cannot reliably produce tool calls. Proving it in isolation first
means every later failure has one fewer possible cause.

A pass here means the model can: pick a tool, fill required arguments, and leave
out an argument it was not given (`size`, for food). A model that fails the third
case will silently invent "large cookie" once it is inside a graph.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from settings import settings  # noqa: E402

ADD_TO_CART_SCHEMA = {
    "type": "function",
    "function": {
        "name": "add_to_cart",
        "description": "Add an item from today's menu to the customer's order.",
        "parameters": {
            "type": "object",
            "properties": {
                "item_name": {"type": "string", "description": "Exact item name."},
                "quantity": {"type": "integer", "description": "How many."},
                "size": {
                    "type": "string",
                    "enum": ["small", "medium", "large"],
                    "description": "Drinks only. Omit entirely for food.",
                },
            },
            "required": ["item_name", "quantity"],
        },
    },
}

SYSTEM = (
    "You are a barista. Today's menu: Latte (drink, $4.00), "
    "Chocolate Chip Cookie (food, $2.00). Drinks have sizes; food does not. "
    "Use the add_to_cart tool when the customer orders something."
)

CASES = [
    ("a large latte please", {"item_name": "Latte", "size": "large"}),
    ("two cookies thanks", {"item_name": "Chocolate Chip Cookie", "quantity": 2}),
]


async def ask(client: httpx.AsyncClient, message: str) -> dict | None:
    response = await client.post(
        "/chat/completions",
        json={
            "model": settings.ollama_model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": message},
            ],
            "tools": [ADD_TO_CART_SCHEMA],
            "temperature": 0,
        },
        timeout=120,
    )
    response.raise_for_status()
    calls = response.json()["choices"][0]["message"].get("tool_calls")
    if not calls:
        return None
    call = calls[0]["function"]
    return {"name": call["name"], "arguments": json.loads(call["arguments"])}


async def main() -> int:
    print(f"model: {settings.ollama_model}")
    print(f"base:  {settings.ollama_base_url}\n")

    failures = 0
    async with httpx.AsyncClient(
        base_url=settings.ollama_base_url, headers={"Authorization": "Bearer ollama"}
    ) as client:
        for message, expected in CASES:
            print(f'  "{message}"')
            try:
                call = await ask(client, message)
            except Exception as error:
                print(f"    ERROR: {error}\n")
                failures += 1
                continue

            if call is None:
                print("    FAIL: no tool call emitted\n")
                failures += 1
                continue

            print(f"    -> {call['name']}({call['arguments']})")
            problems = []
            if call["name"] != "add_to_cart":
                problems.append(f"wrong tool: {call['name']}")
            for key, value in expected.items():
                actual = call["arguments"].get(key)
                if isinstance(value, str) and isinstance(actual, str):
                    if actual.lower() != value.lower():
                        problems.append(f"{key}={actual!r}, expected {value!r}")
                elif actual != value:
                    problems.append(f"{key}={actual!r}, expected {value!r}")

            # The interesting one: food must not acquire a size.
            if "Cookie" in expected["item_name"] and "size" in call["arguments"]:
                problems.append("invented a size for food")

            if problems:
                print(f"    FAIL: {'; '.join(problems)}\n")
                failures += 1
            else:
                print("    ok\n")

    if failures:
        print(f"{failures} case(s) failed — fix the model before touching the graph.")
        print("Try a larger model via OLLAMA_MODEL before assuming your code is wrong.")
        return 1

    print("Tool calling works. Safe to build the graph on this model.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
