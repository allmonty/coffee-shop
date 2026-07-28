"""The domain's public surface (spec §5.3).

Everything outside `shop/` imports from here — `routers/` and, crucially,
`agent/tools.py`, where each tool is named after the service function it calls.
One grep on a tool name therefore reaches the SQL it caused.

The implementation is split by aggregate to keep each file readable; this module
is the map. It has no idea an LLM exists.
"""

from shop.cart import (
    add_to_cart,
    cart_payload,
    change_size,
    get_cart,
    remove_from_cart,
)
from shop.identity import (
    enter,
    normalize_name,
    todays_menu,
    weekday_for,
)
from shop.notes import append_customer_notes
from shop.orders import (
    end_visit,
    get_wallet_balance,
    order_history,
    place_order,
)
from shop.profile import customer_profile

__all__ = [
    # identity and visits
    "enter",
    "normalize_name",
    "todays_menu",
    "weekday_for",
    # cart
    "add_to_cart",
    "cart_payload",
    "change_size",
    "get_cart",
    "remove_from_cart",
    # money
    "end_visit",
    "get_wallet_balance",
    "order_history",
    "place_order",
    # memory
    "append_customer_notes",
    "customer_profile",
]
