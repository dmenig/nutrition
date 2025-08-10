"""add stored logged_date columns with indexes

Revision ID: 20250811_add_logged_date_cols
Revises: 20250810_add_expr_idx
Create Date: 2025-08-11

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "20250811_add_logged_date_cols"
down_revision: Union[str, Sequence[str], None] = "20250810_add_expr_idx"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add nullable date columns
    op.add_column("food_logs", sa.Column("logged_date", sa.Date(), nullable=True))
    op.add_column("sport_activities", sa.Column("logged_date", sa.Date(), nullable=True))

    # Backfill from logged_at
    op.execute("UPDATE food_logs SET logged_date = DATE(logged_at) WHERE logged_date IS NULL")
    op.execute("UPDATE sport_activities SET logged_date = DATE(logged_at) WHERE logged_date IS NULL")

    # Add indexes
    op.create_index("ix_food_logs_logged_date", "food_logs", ["logged_date"], unique=False)
    op.create_index("ix_food_logs_user_logged_date", "food_logs", ["user_id", "logged_date"], unique=False)
    op.create_index("ix_sport_activities_logged_date", "sport_activities", ["logged_date"], unique=False)
    op.create_index(
        "ix_sport_activities_user_logged_date",
        "sport_activities",
        ["user_id", "logged_date"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_sport_activities_user_logged_date", table_name="sport_activities")
    op.drop_index("ix_sport_activities_logged_date", table_name="sport_activities")
    op.drop_index("ix_food_logs_user_logged_date", table_name="food_logs")
    op.drop_index("ix_food_logs_logged_date", table_name="food_logs")
    op.drop_column("sport_activities", "logged_date")
    op.drop_column("food_logs", "logged_date")


