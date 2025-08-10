"""add expression indexes on date(logged_at)

Revision ID: 20250810_add_expr_idx
Revises: 20250810_add_daily_summaries
Create Date: 2025-08-10

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "20250810_add_expr_idx"
down_revision: Union[str, Sequence[str], None] = "20250810_add_daily_summaries"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Expression indexes to accelerate GROUP BY date(logged_at)
    op.execute("CREATE INDEX IF NOT EXISTS ix_food_logs_date ON food_logs ((date(logged_at)))")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_food_logs_user_date ON food_logs (user_id, (date(logged_at)))"
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_sport_activities_date ON sport_activities ((date(logged_at)))")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_sport_activities_user_date ON sport_activities (user_id, (date(logged_at)))"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_sport_activities_user_date")
    op.execute("DROP INDEX IF EXISTS ix_sport_activities_date")
    op.execute("DROP INDEX IF EXISTS ix_food_logs_user_date")
    op.execute("DROP INDEX IF EXISTS ix_food_logs_date")


