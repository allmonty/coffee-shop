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


# add_to_cart exists in two shapes over the SAME service function, one per role
# (spec §13.11). Both are called `add_to_cart` by the model and both grep
# straight to `service.add_to_cart`; they differ only in whether `modifiers` is
# in the schema.
#
# That difference is the barista gate, and it has to be structural for the same
# reason the cashier gate is. Left able to pass extras itself, the waiter simply
# never delegates — measured against the real model, it took "a large espresso
# with oat milk" and handled it alone, and Mo never ran once. A role nothing can
# reach is decoration.


@tool("add_to_cart")
async def add_to_cart(
    item_name: str,
    config: RunnableConfig,
    quantity: int = 1,
    size: str | None = None,
) -> dict:
    """Add a plain item from today's menu to the order.

    size: REQUIRED for drinks, and it must be what the customer actually said —
    ask them if they did not say one. Omit this argument entirely for food;
    passing it for a cookie or pastry is an error.

    This cannot add extras of any kind. If the customer asked for oat or almond
    milk, or an extra shot, send the whole request to ask_barista instead.
    """
    session, visit_id = _context(config)
    return (await service.add_to_cart(session, visit_id, item_name, quantity, size)).to_dict()


@tool("add_to_cart")
async def add_to_cart_with_extras(
    item_name: str,
    config: RunnableConfig,
    quantity: int = 1,
    size: str | None = None,
    modifiers: list[str] | None = None,
) -> dict:
    """Add an item from today's menu to the order.

    size: REQUIRED for drinks, and it must be what the customer actually said.
    Omit this argument entirely for food.

    modifiers: drinks only, and only if the customer asked for one. Use these
    exact codes: "oat_milk", "almond_milk", "extra_shot". At most one milk.
    Plain or regular milk is what the drink already comes with — send no
    modifier for it. Omit this argument entirely otherwise.
    """
    session, visit_id = _context(config)
    return (
        await service.add_to_cart(session, visit_id, item_name, quantity, size, modifiers)
    ).to_dict()


@tool
async def remove_from_cart(
    item_name: str,
    config: RunnableConfig,
    size: str | None = None,
    quantity: int | None = None,
    modifiers: list[str] | None = None,
) -> dict:
    """Remove an item. Omit quantity to take the whole line off.

    size and modifiers only pick between several lines of the same drink — omit
    them unless the order holds that drink more than once.
    """
    session, visit_id = _context(config)
    return (
        await service.remove_from_cart(session, visit_id, item_name, size, quantity, modifiers)
    ).to_dict()


@tool
async def change_size(item_name: str, from_size: str, to_size: str, config: RunnableConfig) -> dict:
    """Resize a drink already in the order, repricing it."""
    session, visit_id = _context(config)
    return (await service.change_size(session, visit_id, item_name, from_size, to_size)).to_dict()


@tool
async def change_modifiers(
    item_name: str,
    config: RunnableConfig,
    to_modifiers: list[str] | None = None,
    size: str | None = None,
    from_modifiers: list[str] | None = None,
) -> dict:
    """Re-do a drink already in the order with different extras, repricing it.

    to_modifiers is the FULL set the drink should end up with, not a change to
    apply — pass an empty list to make it plain. Use these exact codes:
    "oat_milk", "almond_milk", "extra_shot". At most one milk.

    from_modifiers picks which line to change, when the order holds the same
    drink and size more than once. Omit it otherwise.
    """
    session, visit_id = _context(config)
    return (
        await service.change_modifiers(
            session, visit_id, item_name, to_modifiers, size, from_modifiers
        )
    ).to_dict()


# --- the cashier's tools ---------------------------------------------------
#
# Both take NO arguments, and that is the point (spec §13.11).
#
# `confirmed_total_cents` only proves anything because it is the number the model
# that spoke to the customer said out loud. Once a second model sits between the
# waiter and the domain, letting *that* model supply the figure kills the guard
# quietly: the cashier can read the cart, so it would always quote a total that
# matches, and place_order would start rubber-stamping.
#
# So the cashier decides WHETHER to charge; it cannot decide WHAT. Both figures
# are injected by the delegation through config, exactly the way session and
# visit_id already arrive.


def _quoted(config: RunnableConfig) -> int | None:
    return ((config or {}).get("configurable") or {}).get("quoted_total_cents")


def _going_home(config: RunnableConfig) -> bool:
    return bool(((config or {}).get("configurable") or {}).get("going_home"))


@tool
async def charge_the_customer(config: RunnableConfig) -> dict:
    """Charge for the current order at the total the waiter quoted out loud.

    Takes no arguments: the figure is the one the customer already agreed to.
    """
    session, visit_id = _context(config)
    quoted = _quoted(config)
    if quoted is None:
        return {
            "ok": False,
            "error": "nothing_quoted",
            "message": (
                "Nobody has quoted a total to the customer yet, so there is "
                "nothing to charge. Say the total out loud first."
            ),
        }
    return (await service.place_order(session, visit_id, quoted)).to_dict()


@tool
async def send_them_home(config: RunnableConfig) -> dict:
    """End the day and send the customer home. Takes no arguments."""
    session, visit_id = _context(config)
    # Same shape: whether the customer actually said they were leaving is the
    # waiter's observation, not the cashier's guess. The domain still gates it.
    return (await service.end_visit(session, visit_id, _going_home(config))).to_dict()


# The waiter's cart tools. `ask_barista` and `ring_up` are added in
# agent/delegates.py, which cannot be imported here without a cycle.
WAITER_CART_TOOLS = [
    get_menu,
    get_wallet_balance,
    get_cart,
    add_to_cart,
    remove_from_cart,
    change_size,
]

# Drink language -> catalog codes. `change_modifiers` lives here rather than in
# the waiter's list because a sub-agent's schemas never enter the waiter's
# prompt, so it costs nothing on the turns that do not use it.
BARISTA_TOOLS = [get_menu, add_to_cart_with_extras, change_modifiers]

# The end of the visit. get_cart and get_wallet_balance are for reasoning about
# a refusal — never for sourcing the total, which is injected.
CASHIER_READ_TOOLS = [get_cart, get_wallet_balance]


def cashier_tools(*, quoted_total_cents: int | None, going_home: bool) -> list:
    """Val's tools for ONE delegation, scoped to what the waiter authorised.

    Not a fixed list, because a model given a tool will eventually reach for it.
    Val, asked only to take payment, called `send_them_home` as well — the
    domain refused it (nobody had said they were leaving) but the delegation
    then reported failure for a charge that had actually gone through, and Sam
    told the customer their payment had not worked.

    Authority the waiter did not hand over is simply not on the table: no quoted
    total, no way to charge; nobody leaving, no way to close out. Same shape as
    the argument-free tools above — Val decides *whether*, never *what*.
    """
    tools = [*CASHIER_READ_TOOLS]
    if quoted_total_cents is not None:
        tools.append(charge_the_customer)
    if going_home:
        tools.append(send_them_home)
    return tools
