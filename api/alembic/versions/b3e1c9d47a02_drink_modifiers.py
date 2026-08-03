"""drink modifiers

Revision ID: b3e1c9d47a02
Revises: f57fc07f5887
Create Date: 2026-07-29 10:12:03.881204

Line identity becomes (item, size, modifier set), so the old
uq_cart_lines_item_size is replaced by a unique index that includes the new
column. The index uses coalesce(size, '') rather than the bare column: NULLs are
distinct in a unique constraint, so the old one never bound food lines and two
identical croissant rows were legal.

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b3e1c9d47a02'
down_revision: Union[str, None] = 'f57fc07f5887'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

LINE_TABLES = ('cart_lines', 'order_lines')


def upgrade() -> None:
    op.create_table(
        'drink_modifiers',
        sa.Column('code', sa.String(length=20), nullable=False),
        sa.Column('delta_cents', sa.Integer(), nullable=False),
        sa.Column('exclusive_group', sa.String(length=10), nullable=True),
        sa.Column('is_default', sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column('available', sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.CheckConstraint("code ~ '^[a-z_]+$'", name='ck_drink_modifiers_code_format'),
        sa.PrimaryKeyConstraint('code'),
    )

    for table in LINE_TABLES:
        # server_default is what makes a NOT NULL add safe on existing rows, and
        # it is kept permanently so a line can be constructed without naming it.
        op.add_column(
            table,
            sa.Column('modifiers', sa.String(length=120), nullable=False, server_default=''),
        )
        op.create_check_constraint(
            f'ck_{table}_modifiers_drinks_only', table, "sized OR modifiers = ''"
        )
        op.create_check_constraint(
            f'ck_{table}_modifiers_format',
            table,
            "modifiers = '' OR modifiers ~ '^[a-z_]+(,[a-z_]+)*$'",
        )

    op.drop_constraint('uq_cart_lines_item_size', 'cart_lines', type_='unique')
    op.create_index(
        'uq_cart_lines_item_size_modifiers',
        'cart_lines',
        ['cart_id', 'menu_item_id', sa.text("coalesce(size, '')"), 'modifiers'],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index('uq_cart_lines_item_size_modifiers', table_name='cart_lines')
    op.create_unique_constraint(
        'uq_cart_lines_item_size', 'cart_lines', ['cart_id', 'menu_item_id', 'size']
    )

    for table in LINE_TABLES:
        op.drop_constraint(f'ck_{table}_modifiers_format', table, type_='check')
        op.drop_constraint(f'ck_{table}_modifiers_drinks_only', table, type_='check')
        op.drop_column(table, 'modifiers')

    op.drop_table('drink_modifiers')
