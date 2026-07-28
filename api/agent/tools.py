"""Tool wrappers over `shop.service` (spec §7.3).

Two rules make this file boring on purpose:

1. **Each tool is named after the service function it calls.** `add_to_cart` the
   tool calls `service.add_to_cart`. One grep goes from a model's tool call to
   the SQL it caused.
2. **The wrapper holds no logic.** It unpacks config, calls the domain, returns
   the envelope verbatim. Any rule that lived here would be a rule the REST API
   does not enforce, so the same order would behave differently clicked versus
   spoken.

Session and identity arrive through LangGraph's `config["configurable"]` rather
than globals or closures — explicit, greppable, and it avoids a long-lived
closure pinning a request scope.
"""

from __future__ import annotations

import uuid
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from shop import service


def _context(config: RunnableConfig) -> tuple[Any, uuid.UUID]:
    configurable = (config or {}).get("configurable", {})
    session = configurable.get("session")
    visit_id = configurable.get("visit_id")
    if session is None or visit_id is None:
        # Fail loudly. A tool silently operating on nothing is far worse than
        # a stack trace during development.
        raise RuntimeError(
            "tool called without session/visit_id in config['configurable'] — "
            "the caller must inject them when invoking the graph"
        )
    return session, uuid.UUID(str(visit_id))


@tool
async def get_menu(config: RunnableConfig) -> dict:
    """Today's menu with prices. Rarely needed — it is already in your context."""
    session, visit_id = _context(config)
    return {"ok": True, "menu": await service.todays_menu(session, visit_id)}


@tool
async def get_wallet_balance(config: RunnableConfig) -> dict:
    """How much money the customer has left today."""
    session, visit_id = _context(config)
    return (await service.get_wallet_balance(session, visit_id)).to_dict()


@tool
async def get_cart(config: RunnableConfig) -> dict:
    """What is currently in the customer's order, with sizes and the total."""
    session, visit_id = _context(config)
    return (await service.get_cart(session, visit_id)).to_dict()


@tool
async def add_to_cart(
    item_name: str,
    quantity: int,
    config: RunnableConfig,
    size: str | None = None,
) -> dict:
    """Add an item from today's menu. Drinks need a size; food must not have one."""
    session, visit_id = _context(config)
    return (await service.add_to_cart(session, visit_id, item_name, quantity, size)).to_dict()


@tool
async def remove_from_cart(
    item_name: str,
    config: RunnableConfig,
    size: str | None = None,
    quantity: int | None = None,
) -> dict:
    """Remove an item. Omit quantity to take the whole line off."""
    session, visit_id = _context(config)
    return (await service.remove_from_cart(session, visit_id, item_name, size, quantity)).to_dict()


@tool
async def change_size(item_name: str, from_size: str, to_size: str, config: RunnableConfig) -> dict:
    """Resize a drink already in the order, repricing it."""
    session, visit_id = _context(config)
    return (await service.change_size(session, visit_id, item_name, from_size, to_size)).to_dict()


@tool
async def place_order(confirmed_total_cents: int, config: RunnableConfig) -> dict:
    """Charge the customer and hand over the order.

    Pass the exact total you just said out loud. If it does not match the real
    cart total the charge is refused.
    """
    session, visit_id = _context(config)
    return (await service.place_order(session, visit_id, confirmed_total_cents)).to_dict()


@tool
async def end_visit(confirmed: bool, config: RunnableConfig) -> dict:
    """Send the customer home, ending the day. Only after they say they are leaving."""
    session, visit_id = _context(config)
    return (await service.end_visit(session, visit_id, confirmed)).to_dict()


ALL_TOOLS = [
    get_menu,
    get_wallet_balance,
    get_cart,
    add_to_cart,
    remove_from_cart,
    change_size,
    place_order,
    end_visit,
]

TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}
