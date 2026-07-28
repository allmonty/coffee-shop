from sqlalchemy import func, select

from shop.catalog_data import DRINKS, FOODS, SIZE_DELTAS
from shop.models import MenuItem, SizeModifier
from shop.seed import seed_catalog


async def test_seed_inserts_the_whole_catalog(session):
    added = await seed_catalog(session)

    assert added == len(DRINKS) + len(FOODS) + len(SIZE_DELTAS)

    drinks = await session.scalar(
        select(func.count()).select_from(MenuItem).where(MenuItem.category == "drink")
    )
    foods = await session.scalar(
        select(func.count()).select_from(MenuItem).where(MenuItem.category == "food")
    )
    assert (drinks, foods) == (17, 13)


async def test_seed_is_idempotent(session):
    await seed_catalog(session)
    added_again = await seed_catalog(session)

    assert added_again == 0
    total = await session.scalar(select(func.count()).select_from(MenuItem))
    assert total == len(DRINKS) + len(FOODS)


async def test_prices_match_the_spec(session):
    await seed_catalog(session)

    for name, expected_cents in DRINKS + FOODS:
        item = await session.scalar(select(MenuItem).where(MenuItem.name == name))
        assert item is not None, f"{name} missing from catalog"
        assert item.price_cents == expected_cents, name


async def test_only_drinks_are_sized(session):
    await seed_catalog(session)

    items = (await session.scalars(select(MenuItem))).all()
    for item in items:
        assert item.sized == (item.category == "drink"), item.name


async def test_size_deltas_are_flat_and_ordered(session):
    await seed_catalog(session)

    deltas = {m.size: m.delta_cents for m in (await session.scalars(select(SizeModifier))).all()}

    assert deltas == {"small": 0, "medium": 60, "large": 120}


async def test_seed_leaves_edited_prices_alone(session):
    """The seed establishes the catalog; it does not own prices afterwards."""
    await seed_catalog(session)
    latte = await session.scalar(select(MenuItem).where(MenuItem.name == "Latte"))
    latte.price_cents = 999
    await session.commit()

    await seed_catalog(session)

    latte = await session.scalar(select(MenuItem).where(MenuItem.name == "Latte"))
    assert latte.price_cents == 999


async def test_catalog_has_cheap_anchors_for_the_daily_draw(session):
    """Guards the precondition the §3.2 generator depends on.

    If someone prices the cheap end of the catalog out of reach, the daily menu
    can no longer guarantee an affordable day. Failing here says why.
    """
    await seed_catalog(session)

    cheap_drinks = await session.scalar(
        select(func.count())
        .select_from(MenuItem)
        .where(MenuItem.category == "drink", MenuItem.price_cents <= 300)
    )
    cheap_foods = await session.scalar(
        select(func.count())
        .select_from(MenuItem)
        .where(MenuItem.category == "food", MenuItem.price_cents <= 250)
    )

    assert cheap_drinks >= 1
    assert cheap_foods >= 1
