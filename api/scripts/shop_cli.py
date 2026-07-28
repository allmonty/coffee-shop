"""Talk to the barista in a terminal.

    uv run python scripts/shop_cli.py --name Allan

This is the debugging surface the project is built around. A terminal beats a
chat window for developing an agent because the whole transcript stays in the
scrollback next to the trace — you can see the tool calls, the totals and the
recovery in one screen.

Type `/cart`, `/wallet` or `/state` to inspect without talking to the model.
Ctrl-D or `/quit` leaves without ending the in-game day.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.checkpointer import open_checkpointer  # noqa: E402
from agent.graph import build_graph  # noqa: E402
from agent.runner import run_turn  # noqa: E402
from db import SessionLocal, engine  # noqa: E402
from settings import settings  # noqa: E402
from shop import service  # noqa: E402
from shop.pricing import format_cents  # noqa: E402
from shop.seed import seed_catalog  # noqa: E402

DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


async def converse(name: str) -> None:
    async with SessionLocal() as session:
        await seed_catalog(session)
        entered = await service.enter(session, name)
        if not entered.ok:
            print(f"{entered.message}")
            return

        data = entered.data
        visit_id = uuid.UUID(data["visit_id"])
        user_id = uuid.UUID(data["user_id"])

        print(f"\n{BOLD}The Coffee Shop{RESET}")
        print(f"{DIM}model: {settings.ollama_model}{RESET}")
        print(
            f"{DIM}day {data['day']} ({data['weekday']}) · "
            f"wallet {format_cents(data['wallet_cents'])} · "
            f"{len(data['menu'])} items on today's menu{RESET}\n"
        )

        async with open_checkpointer() as checkpointer:
            graph = build_graph(checkpointer=checkpointer)

            day = data["day"]
            await turn(session, graph, user_id, visit_id, day=day, event="on_enter")

            while True:
                try:
                    text = input(f"{BOLD}you>{RESET} ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break

                if not text:
                    continue
                if text in {"/quit", "/exit"}:
                    break
                if text == "/cart":
                    print(f"{DIM}{(await service.get_cart(session, visit_id)).to_dict()}{RESET}\n")
                    continue
                if text == "/wallet":
                    balance = await service.get_wallet_balance(session, visit_id)
                    print(f"{DIM}{balance.to_dict()}{RESET}\n")
                    continue
                if text == "/state":
                    snapshot = await graph.aget_state(
                        {"configurable": {"thread_id": str(visit_id)}}
                    )
                    print(f"{DIM}{len(snapshot.values.get('messages', []))} messages{RESET}\n")
                    continue

                ended = await turn(session, graph, user_id, visit_id, day=day, message=text)
                if ended:
                    print(f"{DIM}— next morning —{RESET}\n")
                    break

    await engine.dispose()


async def turn(session, graph, user_id, visit_id, **kwargs) -> bool:
    """Stream one turn, printing tokens as they arrive."""
    print(f"{BOLD}sam>{RESET} ", end="", flush=True)
    ended = False
    saw_text = False

    async for frame in run_turn(
        session=session, graph=graph, user_id=user_id, visit_id=visit_id, **kwargs
    ):
        if frame["type"] == "token":
            print(frame["text"], end="", flush=True)
            saw_text = True
        elif frame["type"] == "cart_updated" and frame.get("lines"):
            total = format_cents(frame.get("total_cents", 0))
            print(f"\n{DIM}   [cart: {len(frame['lines'])} line(s), {total}]{RESET}", end="")
        elif frame["type"] == "visit_ended":
            ended = True
        elif frame["type"] == "error":
            print(f"\n{DIM}[error: {frame.get('error')} {frame.get('detail', '')}]{RESET}", end="")

    if not saw_text:
        print(f"{DIM}(no reply){RESET}", end="")
    print("\n")
    return ended


def main() -> None:
    parser = argparse.ArgumentParser(description="Talk to the barista.")
    parser.add_argument("--name", default="Allan", help="Customer name (the whole identity).")
    args = parser.parse_args()
    asyncio.run(converse(args.name))


if __name__ == "__main__":
    main()
