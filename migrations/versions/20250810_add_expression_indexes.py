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
    # No-op; superseded by adding stored date columns with normal btree indexes
    pass


def downgrade() -> None:
    pass


