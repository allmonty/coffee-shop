"""The catalog, transcribed from spec §3.1, and the size deltas from §3.4.

Kept as data rather than SQL so the seed script and the tests can both assert
against one source. The price spread ($1.75–$6.50) is deliberate: it is wide
enough that a careless daily draw could produce an all-expensive day, which is
what makes the §3.2 affordability guarantees load-bearing rather than
decorative.
"""

DRINKS: list[tuple[str, int]] = [
    ("Filter Coffee", 175),
    ("Espresso", 200),
    ("Americano", 250),
    ("Doppio", 275),
    ("Macchiato", 300),
    ("Cortado", 325),
    ("Latte", 400),
    ("Hot Chocolate", 400),
    ("Flat White", 425),
    ("Cappuccino", 450),
    ("Iced Latte", 450),
    ("Chai Latte", 475),
    ("Cold Brew", 475),
    ("Mocha", 500),
    ("Caramel Latte", 525),
    ("Matcha Latte", 550),
    ("Affogato", 650),
]

FOODS: list[tuple[str, int]] = [
    ("Shortbread", 175),
    ("Chocolate Chip Cookie", 200),
    ("Oatmeal Cookie", 200),
    ("Banana Bread", 300),
    ("Blueberry Muffin", 325),
    ("Croissant", 350),
    ("Brownie", 375),
    ("Pain au Chocolat", 400),
    ("Almond Croissant", 425),
    ("Bagel & Cream Cheese", 425),
    ("Cinnamon Roll", 450),
    ("Carrot Cake", 500),
    ("Cheesecake Slice", 550),
]

# Flat across every drink, so one three-row table covers size pricing instead of
# a price row per item per size (spec §3.4).
SIZE_DELTAS: dict[str, int] = {
    "small": 0,
    "medium": 60,
    "large": 120,
}
