"""The cart: what the customer has asked for but not yet paid for.

Every error here returns a `message` written to be read aloud. `size_required`
in particular is not a failure — it is how the domain tells the agent the
customer's request was incomplete, so the barista asks instead of guessing
(spec §6.4).

Modifiers are the mirror image of that, and the asymmetry is worth naming:
there is no `modifier_required`, because a drink with no modifiers is a complete
order, not an incomplete one. A modifier request can only ever be
over-specified-and-unrecognized (`unknown_modifier`, `modifier_conflict`), never
under-specified — which is why the two axes need different error shapes even
though they otherwise look alike.
"""

from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from shop.models import SIZES, Cart, CartLine, MenuItem, Visit, VisitMenuItem
from shop.modifiers import canonical_key, describe, parse_key
from shop.pricing import format_cents, load_deltas, unit_price_cents
from shop.result import Result


async def open_visit(session: AsyncSession, visit_id: uuid.UUID) -> Visit | None:
    visit = await session.get(Visit, visit_id)
    if visit is None or visit.ended_at is not None:
        return None
    return visit


async def _cart_for(session: AsyncSession, visit_id: uuid.UUID) -> Cart:
    cart = await session.scalar(select(Cart).where(Cart.visit_id == visit_id))
    if cart is None:
        cart = Cart(visit_id=visit_id)
        session.add(cart)
        await session.flush()
    return cart


async def _find_in_catalog(session: AsyncSession, item_name: str) -> MenuItem | None:
    """Case-insensitive exact match. The model types what the customer said."""
    return await session.scalar(
        select(MenuItem).where(
            func.lower(MenuItem.name) == item_name.strip().lower(),
            MenuItem.in_catalog.is_(True),
        )
    )


async def _is_on_todays_menu(session: AsyncSession, visit_id: uuid.UUID, item_id: int) -> bool:
    found = await session.scalar(
        select(VisitMenuItem.menu_item_id).where(
            VisitMenuItem.visit_id == visit_id,
            VisitMenuItem.menu_item_id == item_id,
        )
    )
    return found is not None


async def cart_payload(session: AsyncSession, visit_id: uuid.UUID) -> dict[str, object]:
    """Lines with sizes and prices, plus the total (spec §7.2 `cart_updated`)."""
    cart = await session.scalar(select(Cart).where(Cart.visit_id == visit_id))
    if cart is None:
        return {"lines": [], "total_cents": 0}

    deltas = await load_deltas(session)
    rows = (
        await session.execute(
            select(CartLine, MenuItem)
            .join(MenuItem, MenuItem.id == CartLine.menu_item_id)
            .where(CartLine.cart_id == cart.id)
            .order_by(CartLine.id)
        )
    ).all()

    lines = []
    total = 0
    for line, item in rows:
        unit = unit_price_cents(item, line.size, line.modifiers, deltas)
        line_total = unit * line.quantity
        total += line_total
        lines.append(
            {
                "item": item.name,
                "size": line.size,
                "modifiers": list(parse_key(line.modifiers)),
                "quantity": line.quantity,
                "unit_price_cents": unit,
                "line_total_cents": line_total,
            }
        )
    return {"lines": lines, "total_cents": total}


async def get_cart(session: AsyncSession, visit_id: uuid.UUID) -> Result:
    if await open_visit(session, visit_id) is None:
        return _visit_closed()
    return Result.success(**await cart_payload(session, visit_id))


async def add_to_cart(
    session: AsyncSession,
    visit_id: uuid.UUID,
    item_name: str,
    quantity: int = 1,
    size: str | None = None,
    modifiers: list[str] | None = None,
) -> Result:
    """Six of these failures are distinct truths, not one generic rejection.

    `unknown_item` / `not_available_today` / `size_required` / `size_not_applicable`
    / `unknown_modifier` / `modifier_conflict` each need a different sentence
    from the barista. The rest — `visit_closed`, `invalid_quantity`,
    `unknown_size`, `modifier_not_applicable` — are mechanical.

    The size checks deliberately run before any modifier check. `size_required`
    is the only branch here that asks a *question* rather than reporting a
    correction, so it has to reach the barista first: "a latte with soy" should
    come back "which size?" and let the model retry knowing both facts.
    """
    if await open_visit(session, visit_id) is None:
        return _visit_closed()

    if quantity < 1:
        return Result.failure("invalid_quantity", "How many would you like?")

    item = await _find_in_catalog(session, item_name)
    if item is None:
        return Result.failure(
            "unknown_item",
            f"We don't do {item_name}, I'm afraid.",
        )

    if not await _is_on_todays_menu(session, visit_id, item.id):
        # Real item, just not drawn today. "We're not doing mochas today" is
        # true; "we don't sell mochas" is not (spec §3.3).
        return Result.failure(
            "not_available_today",
            f"No {item.name} today, sorry — it's not on the board.",
        )

    if item.sized and size is None:
        return Result.failure("size_required", "Which size — small, medium, or large?")
    if not item.sized and size is not None:
        return Result.failure(
            "size_not_applicable",
            f"{item.name} only comes the one size.",
        )
    if size is not None and size not in SIZES:
        return Result.failure("unknown_size", "We do small, medium and large.")

    deltas = await load_deltas(session)
    if not item.sized and modifiers:
        return Result.failure(
            "modifier_not_applicable",
            f"{item.name} doesn't take any extras, I'm afraid.",
        )

    key = canonical_key(modifiers, deltas.modifiers)
    offerable = deltas.offerable()
    for code in parse_key(key):
        if code not in offerable:
            return Result.failure(
                "unknown_modifier",
                f"We don't do {code.replace('_', ' ')}, sorry — there's "
                f"{_offer_list(offerable)}. Which would you like?",
            )

    conflict = _conflicting_group(parse_key(key), deltas)
    if conflict is not None:
        first, second = conflict
        return Result.failure(
            "modifier_conflict",
            f"{first.replace('_', ' ').title()} or {second.replace('_', ' ')} — "
            "can't do both in one cup. Which one?",
        )

    cart = await _cart_for(session, visit_id)
    line = await session.scalar(
        select(CartLine).where(
            CartLine.cart_id == cart.id,
            CartLine.menu_item_id == item.id,
            CartLine.size.is_(None) if size is None else CartLine.size == size,
            CartLine.modifiers == key,
        )
    )
    if line is None:
        session.add(
            CartLine(
                cart_id=cart.id,
                menu_item_id=item.id,
                quantity=quantity,
                size=size,
                sized=item.sized,
                modifiers=key,
            )
        )
    else:
        line.quantity += quantity

    cart.version += 1
    await session.flush()

    payload = await cart_payload(session, visit_id)
    await session.commit()
    return Result.success(
        f"Added {quantity} {_describe(item.name, size, key)}.",
        added=item.name,
        size=size,
        modifiers=list(parse_key(key)),
        quantity=quantity,
        **payload,
    )


async def remove_from_cart(
    session: AsyncSession,
    visit_id: uuid.UUID,
    item_name: str,
    size: str | None = None,
    quantity: int | None = None,
    modifiers: list[str] | None = None,
) -> Result:
    """`quantity=None` removes the whole line.

    `size` and `modifiers` are *filters*, used only to pick between several
    lines of the same item. Note the asymmetry with `add_to_cart`: there an
    empty `modifiers` means "no extras", here it means "I have nothing to say
    about extras". A model that sends `[]` almost always means the latter, and
    treating it as "the plain one" would quietly delete the oat latte when the
    customer asked for the plain one to go.
    """
    if await open_visit(session, visit_id) is None:
        return _visit_closed()

    item = await _find_in_catalog(session, item_name)
    if item is None:
        return Result.failure("unknown_item", f"We don't do {item_name}, I'm afraid.")

    cart = await _cart_for(session, visit_id)
    matches = (
        await session.scalars(
            select(CartLine).where(CartLine.cart_id == cart.id, CartLine.menu_item_id == item.id)
        )
    ).all()
    matches = [line for line in matches if size is None or line.size == size]
    if modifiers:
        deltas = await load_deltas(session)
        key = canonical_key(modifiers, deltas.modifiers)
        matches = [line for line in matches if line.modifiers == key]

    if not matches:
        return Result.failure("not_in_cart", f"There's no {item.name} in the order.")
    if len(matches) > 1:
        # Size first when both axes are ambiguous: it is the coarser question,
        # and the style rule is one question at a time.
        if len({line.size for line in matches}) > 1:
            sizes = ", ".join(sorted(line.size or "" for line in matches))
            return Result.failure(
                "size_ambiguous",
                f"You've got {item.name} in two sizes ({sizes}) — which one?",
            )
        variants = " or the ".join(_variant(line.modifiers) for line in matches)
        return Result.failure(
            "modifier_ambiguous",
            f"You've got two {item.name}s there — the {variants} one?",
        )

    line = matches[0]
    if quantity is None or quantity >= line.quantity:
        await session.delete(line)
    else:
        line.quantity -= quantity

    cart.version += 1
    await session.flush()

    payload = await cart_payload(session, visit_id)
    await session.commit()
    return Result.success(
        f"Took the {_describe(item.name, line.size, line.modifiers)} off.",
        removed=item.name,
        size=line.size,
        modifiers=list(parse_key(line.modifiers)),
        **payload,
    )


async def change_size(
    session: AsyncSession,
    visit_id: uuid.UUID,
    item_name: str,
    from_size: str,
    to_size: str,
) -> Result:
    """The size-upsell path in one call, so it reads as one step in the trace."""
    if await open_visit(session, visit_id) is None:
        return _visit_closed()

    if to_size not in SIZES:
        return Result.failure("unknown_size", "We do small, medium and large.")

    item = await _find_in_catalog(session, item_name)
    if item is None:
        return Result.failure("unknown_item", f"We don't do {item_name}, I'm afraid.")
    if not item.sized:
        return Result.failure("size_not_applicable", f"{item.name} only comes the one size.")

    cart = await _cart_for(session, visit_id)
    # .all(), not .scalar(): scalar() returns the first row without complaining
    # about the rest, so once the same drink can exist in several modifier
    # variants at one size, this would silently resize an arbitrary one.
    candidates = (
        await session.scalars(
            select(CartLine).where(
                CartLine.cart_id == cart.id,
                CartLine.menu_item_id == item.id,
                CartLine.size == from_size,
            )
        )
    ).all()
    if not candidates:
        return Result.failure(
            "not_in_cart",
            f"There's no {from_size} {item.name} in the order.",
        )
    if len(candidates) > 1:
        variants = " or the ".join(_variant(line.modifiers) for line in candidates)
        return Result.failure(
            "modifier_ambiguous",
            f"You've got two {from_size} {item.name}s there — the {variants} one?",
        )
    line = candidates[0]

    existing = await session.scalar(
        select(CartLine).where(
            CartLine.cart_id == cart.id,
            CartLine.menu_item_id == item.id,
            CartLine.size == to_size,
            # Same modifiers, or this merges an oat latte into a plain one and
            # charges the plain price for it.
            CartLine.modifiers == line.modifiers,
        )
    )
    if existing is None:
        line.size = to_size
    else:
        # Merging keeps the unique (cart, item, size, modifiers) index satisfiable.
        existing.quantity += line.quantity
        await session.delete(line)

    cart.version += 1
    await session.flush()

    deltas = await load_deltas(session)
    # The modifier surcharge is identical on both sides and cancels, so the
    # quoted difference stays the pure size difference the barista said aloud.
    difference = unit_price_cents(item, to_size, line.modifiers, deltas) - unit_price_cents(
        item, from_size, line.modifiers, deltas
    )
    payload = await cart_payload(session, visit_id)
    await session.commit()
    return Result.success(
        f"Made it a {to_size} {item.name}, {format_cents(abs(difference))} "
        f"{'more' if difference >= 0 else 'less'}.",
        item=item.name,
        from_size=from_size,
        to_size=to_size,
        difference_cents=difference,
        **payload,
    )


async def change_modifiers(
    session: AsyncSession,
    visit_id: uuid.UUID,
    item_name: str,
    to_modifiers: list[str] | None = None,
    size: str | None = None,
    from_modifiers: list[str] | None = None,
) -> Result:
    """Re-do a drink already in the order with different extras, repricing it.

    The modifier twin of `change_size`, and it exists for the same reason: one
    step in the trace instead of a remove-then-re-add whose middle state is a
    cart the customer never asked for.

    `to_modifiers` describes the RESULT, so an empty list means "make it plain"
    — unlike `remove_from_cart`, where an empty list means "I have nothing to
    say about extras". `from_modifiers` picks WHICH line to change, and is the
    twin of `change_size`'s `from_size`: without it, a cart holding the same
    drink twice at one size can only answer `modifier_ambiguous`, which also
    made the merge below unreachable.
    """
    if await open_visit(session, visit_id) is None:
        return _visit_closed()

    item = await _find_in_catalog(session, item_name)
    if item is None:
        return Result.failure("unknown_item", f"We don't do {item_name}, I'm afraid.")
    if not item.sized:
        return Result.failure(
            "modifier_not_applicable",
            f"{item.name} doesn't take any extras, I'm afraid.",
        )

    deltas = await load_deltas(session)
    key = canonical_key(to_modifiers, deltas.modifiers)
    offerable = deltas.offerable()
    for code in parse_key(key):
        if code not in offerable:
            return Result.failure(
                "unknown_modifier",
                f"We don't do {code.replace('_', ' ')}, sorry — there's "
                f"{_offer_list(offerable)}. Which would you like?",
            )
    conflict = _conflicting_group(parse_key(key), deltas)
    if conflict is not None:
        first, second = conflict
        return Result.failure(
            "modifier_conflict",
            f"{first.replace('_', ' ').title()} or {second.replace('_', ' ')} — "
            "can't do both in one cup. Which one?",
        )

    cart = await _cart_for(session, visit_id)
    candidates = (
        await session.scalars(
            select(CartLine).where(
                CartLine.cart_id == cart.id,
                CartLine.menu_item_id == item.id,
                *([CartLine.size == size] if size else []),
            )
        )
    ).all()
    if from_modifiers is not None:
        from_key = canonical_key(from_modifiers, deltas.modifiers)
        candidates = [line for line in candidates if line.modifiers == from_key]
    if not candidates:
        return Result.failure("not_in_cart", f"There's no {item.name} in the order.")
    if len(candidates) > 1:
        if len({line.size for line in candidates}) > 1:
            sizes = ", ".join(sorted(line.size or "" for line in candidates))
            return Result.failure(
                "size_ambiguous",
                f"You've got {item.name} in two sizes ({sizes}) — which one?",
            )
        variants = " or the ".join(_variant(line.modifiers) for line in candidates)
        return Result.failure(
            "modifier_ambiguous",
            f"You've got two {item.name}s there — the {variants} one?",
        )

    line = candidates[0]
    before = unit_price_cents(item, line.size, line.modifiers, deltas)
    existing = await session.scalar(
        select(CartLine).where(
            CartLine.cart_id == cart.id,
            CartLine.menu_item_id == item.id,
            CartLine.size == line.size,
            CartLine.modifiers == key,
            CartLine.id != line.id,
        )
    )
    if existing is None:
        line.modifiers = key
    else:
        existing.quantity += line.quantity
        await session.delete(line)

    cart.version += 1
    await session.flush()

    difference = unit_price_cents(item, line.size, key, deltas) - before
    payload = await cart_payload(session, visit_id)
    await session.commit()
    return Result.success(
        f"Made it {_describe(item.name, line.size, key)}, "
        f"{format_cents(abs(difference))} {'more' if difference >= 0 else 'less'}.",
        item=item.name,
        size=line.size,
        modifiers=list(parse_key(key)),
        difference_cents=difference,
        **payload,
    )


def _describe(item_name: str, size: str | None, modifiers: str = "") -> str:
    described = f"{size} {item_name}" if size else item_name
    return f"{described} with {describe(modifiers)}" if modifiers else described


def _variant(modifiers: str) -> str:
    """How to name one line when picking between two of the same drink."""
    return describe(modifiers) if modifiers else "plain"


def _offer_list(offerable: dict[str, int]) -> str:
    """`oat milk, almond milk or an extra shot`, for read-aloud error messages."""
    names = [code.replace("_", " ") for code in sorted(offerable)]
    if not names:
        return "nothing extra today"
    if len(names) == 1:
        return names[0]
    return f"{', '.join(names[:-1])} or {names[-1]}"


def _conflicting_group(codes: tuple[str, ...], deltas) -> tuple[str, str] | None:
    """The first pair of codes that cannot share a cup, if any."""
    seen: dict[str, str] = {}
    for code in codes:
        group = deltas.modifiers[code].exclusive_group
        if group is None:
            continue
        if group in seen:
            return seen[group], code
        seen[group] = code
    return None


def _visit_closed() -> Result:
    return Result.failure("visit_closed", "That visit's already finished.")
