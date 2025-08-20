"""
add more food nutrients columns: sfat, free_sugar, fibres

Revision ID: 20250813_add_more_food_nutrients
Revises: 20250812_add_weight_logs
Create Date: 2025-08-13 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "20250813_add_more_food_nutrients"
down_revision: Union[str, None] = "20250812_add_weight_logs"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("foods", schema=None) as batch_op:
        batch_op.add_column(sa.Column("sfat", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("free_sugar", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("fibres", sa.Float(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("foods", schema=None) as batch_op:
        batch_op.drop_column("fibres")
        batch_op.drop_column("free_sugar")
        batch_op.drop_column("sfat")


