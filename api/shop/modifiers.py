"""Turning a list of modifier codes into the one string that identifies a line.

`cart_lines.modifiers` is what makes two differently-modified lattes two lines
and the same latte ordered twice one line of quantity 2. That only works if the
same set of codes always produces byte-identical text, so every write goes
through `canonical_key` and nothing else builds the string by hand.

This is the single rule in the modifier design that the database cannot enforce
for us (spec §13, decision 8): the schema guarantees the column's *shape* with a
CHECK, and guarantees uniqueness over it, but only this module guarantees that
["oat_milk", "extra_shot"] and ["extra_shot", "oat_milk"] mean the same cup.
Hence its own module and its own tests.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from shop.models import DrinkModifier

SEPARATOR = ","


def canonical_key(
    codes: Iterable[str] | None,
    catalog: Mapping[str, DrinkModifier],
) -> str:
    """The line-identity string for a set of modifier codes.

    Lowercased, whitespace stripped, defaults dropped, deduped, sorted, joined.
    Unknown codes are kept rather than dropped — validation is `add_to_cart`'s
    job, and silently discarding a code the customer asked for would charge them
    for a drink they did not order.
    """
    if not codes:
        return ""

    cleaned = set()
    for raw in codes:
        code = raw.strip().lower().replace(" ", "_").replace("-", "_")
        if not code:
            continue
        # whole milk is the drink as listed, so it carries no identity: a latte
        # and a latte with regular milk have to land on the same cart line.
        modifier = catalog.get(code)
        if modifier is not None and modifier.is_default:
            continue
        cleaned.add(code)

    return SEPARATOR.join(sorted(cleaned))


def parse_key(key: str) -> tuple[str, ...]:
    """The codes in a canonical key, in the order they are stored."""
    if not key:
        return ()
    return tuple(key.split(SEPARATOR))


def describe(key: str) -> str:
    """`oat milk, extra shot`. For messages the barista reads aloud."""
    return ", ".join(code.replace("_", " ") for code in parse_key(key))
