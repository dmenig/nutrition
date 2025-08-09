"""add carried_weight_kg and distance_m to sport_activities

Revision ID: 20250809_add_sport_cols
Revises: 20250809_add_indexes
Create Date: 2025-08-09

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "20250809_add_sport_cols"
down_revision: Union[str, Sequence[str], None] = "20250809_add_indexes"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "sport_activities",
        sa.Column("carried_weight_kg", sa.Float(), nullable=True),
    )
    op.add_column(
        "sport_activities",
        sa.Column("distance_m", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("sport_activities", "distance_m")
    op.drop_column("sport_activities", "carried_weight_kg")


