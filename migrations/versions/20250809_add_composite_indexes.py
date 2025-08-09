"""add composite indexes for date queries

Revision ID: 20250809_add_indexes
Revises: 91df63e49043
Create Date: 2025-08-09

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "20250809_add_indexes"
down_revision: Union[str, Sequence[str], None] = "91df63e49043"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # New columns for sport details
    op.add_column("sport_activities", sa.Column("carried_weight_kg", sa.Float(), nullable=True))
    op.add_column("sport_activities", sa.Column("distance_m", sa.Float(), nullable=True))
    op.create_index(
        "ix_food_logs_user_id_logged_at",
        "food_logs",
        ["user_id", "logged_at"],
        unique=False,
    )
    op.create_index(
        "ix_sport_activities_user_id_logged_at",
        "sport_activities",
        ["user_id", "logged_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_column("sport_activities", "distance_m")
    op.drop_column("sport_activities", "carried_weight_kg")
    op.drop_index("ix_sport_activities_user_id_logged_at", table_name="sport_activities")
    op.drop_index("ix_food_logs_user_id_logged_at", table_name="food_logs")


