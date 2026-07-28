"""SQLAlchemy tables (spec §8).

Money is integer cents everywhere. `order_lines` snapshots the unit price it
actually charged, including any size surcharge, so historical orders stay
correct when the catalog or `size_modifiers` is edited.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

DRINK = "drink"
FOOD = "food"
SIZES = ("small", "medium", "large")


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # Display name exactly as typed; name_key is the lookup key (spec §4.1).
    name: Mapped[str] = mapped_column(String(40))
    name_key: Mapped[str] = mapped_column(String(40), unique=True, index=True)
    current_day: Mapped[int] = mapped_column(Integer, default=1)
    wallet_cents: Mapped[int] = mapped_column(Integer, default=2000)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class MenuItem(Base):
    """The catalog: everything the shop can ever serve (spec §3.1)."""

    __tablename__ = "menu_items"
    __table_args__ = (
        CheckConstraint(f"category IN ('{DRINK}', '{FOOD}')", name="ck_menu_items_category"),
        # Drinks are sized, food is not. Encoded here so the rule cannot drift
        # away from the line constraints that depend on it (spec §3.4).
        CheckConstraint(
            f"(category = '{DRINK}') = sized",
            name="ck_menu_items_only_drinks_are_sized",
        ),
        # Redundant on its own — id is already unique — but it is the target a
        # composite FK needs so cart_lines and order_lines can constrain size
        # against this item's `sized` flag. See CartLine.
        UniqueConstraint("id", "sized", name="uq_menu_items_id_sized"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(60), unique=True)
    category: Mapped[str] = mapped_column(String(10))
    price_cents: Mapped[int] = mapped_column(Integer)
    sized: Mapped[bool] = mapped_column(Boolean)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Retire an item without orphaning the orders that reference it.
    in_catalog: Mapped[bool] = mapped_column(Boolean, default=True)


class SizeModifier(Base):
    """Three rows. Keeps size pricing in the database like every other price."""

    __tablename__ = "size_modifiers"
    __table_args__ = (
        CheckConstraint("size IN ('small', 'medium', 'large')", name="ck_size_modifiers_size"),
    )

    size: Mapped[str] = mapped_column(String(10), primary_key=True)
    delta_cents: Mapped[int] = mapped_column(Integer)


class Visit(Base):
    __tablename__ = "visits"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    day: Mapped[int] = mapped_column(Integer)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    menu_items: Mapped[list[VisitMenuItem]] = relationship(cascade="all, delete-orphan")


class VisitMenuItem(Base):
    """Today's menu: the subset drawn for this visit (spec §3.2).

    The whole implementation of the daily menu is this join table, written once
    when the visit opens. "Not on today's menu" is therefore a join, not a rule
    someone has to remember to apply.
    """

    __tablename__ = "visit_menu_items"

    visit_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("visits.id", ondelete="CASCADE"), primary_key=True
    )
    menu_item_id: Mapped[int] = mapped_column(ForeignKey("menu_items.id"), primary_key=True)


class Cart(Base):
    __tablename__ = "carts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    visit_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("visits.id", ondelete="CASCADE"))
    # Supports idempotent place_order (spec §6.4).
    version: Mapped[int] = mapped_column(Integer, default=1)


class CartLine(Base):
    """A line in the cart.

    Making "large cookie" unrepresentable needs a rule spanning two tables, which
    a plain CHECK cannot express. The standard relational answer, used here:
    carry the item's `sized` flag on the line, tie it to the catalog with a
    composite foreign key so it cannot disagree, then CHECK size against it
    locally. The result is that the database rejects a sized food line and a
    sizeless drink line — no trigger, no application-layer trust (spec §8).
    """

    __tablename__ = "cart_lines"
    __table_args__ = (
        ForeignKeyConstraint(
            ["menu_item_id", "sized"],
            ["menu_items.id", "menu_items.sized"],
            name="fk_cart_lines_item_sized",
        ),
        CheckConstraint(
            "size IS NULL OR size IN ('small', 'medium', 'large')",
            name="ck_cart_lines_size_value",
        ),
        CheckConstraint("sized = (size IS NOT NULL)", name="ck_cart_lines_size_matches_item"),
        UniqueConstraint("cart_id", "menu_item_id", "size", name="uq_cart_lines_item_size"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cart_id: Mapped[int] = mapped_column(ForeignKey("carts.id", ondelete="CASCADE"))
    menu_item_id: Mapped[int] = mapped_column(Integer)
    quantity: Mapped[int] = mapped_column(Integer)
    # NULL means "this item has no size", which is a different fact from small.
    size: Mapped[str | None] = mapped_column(String(10), nullable=True)
    sized: Mapped[bool] = mapped_column(Boolean)


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    visit_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("visits.id", ondelete="CASCADE"))
    day: Mapped[int] = mapped_column(Integer)
    total_cents: Mapped[int] = mapped_column(Integer)
    placed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class OrderLine(Base):
    __tablename__ = "order_lines"
    __table_args__ = (
        ForeignKeyConstraint(
            ["menu_item_id", "sized"],
            ["menu_items.id", "menu_items.sized"],
            name="fk_order_lines_item_sized",
        ),
        CheckConstraint(
            "size IS NULL OR size IN ('small', 'medium', 'large')",
            name="ck_order_lines_size_value",
        ),
        CheckConstraint("sized = (size IS NOT NULL)", name="ck_order_lines_size_matches_item"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("orders.id", ondelete="CASCADE"))
    menu_item_id: Mapped[int] = mapped_column(Integer)
    quantity: Mapped[int] = mapped_column(Integer)
    size: Mapped[str | None] = mapped_column(String(10), nullable=True)
    sized: Mapped[bool] = mapped_column(Boolean)
    # Snapshot of base + size delta at the time of the order.
    unit_price_cents: Mapped[int] = mapped_column(Integer)


class Message(Base):
    """The application's own transcript, for display and history.

    Separate from the LangGraph checkpointer on purpose: the checkpointer's
    format is LangGraph's business and will change under you (spec §8).
    """

    __tablename__ = "messages"
    __table_args__ = (
        CheckConstraint("role IN ('user', 'barista', 'tool')", name="ck_messages_role"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    visit_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("visits.id", ondelete="CASCADE"))
    role: Mapped[str] = mapped_column(String(10))
    content: Mapped[str] = mapped_column(Text)
    tool_name: Mapped[str | None] = mapped_column(String(40), nullable=True)
    inserted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class CustomerPreference(Base):
    """ONLY model-written notes (spec §6.5).

    favorite_drink / favorite_food / usual_order / visit_count / last_visit_day
    are NOT stored — they are aggregated from orders and visits at read time.
    """

    __tablename__ = "customer_preferences"

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    notes: Mapped[list] = mapped_column(JSONB, default=list)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
