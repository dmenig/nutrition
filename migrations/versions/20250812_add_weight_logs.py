"""add weight_logs table

Revision ID: 20250812_add_weight_logs
Revises: 20250811_add_logged_date_columns
Create Date: 2025-08-17 00:00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '20250812_add_weight_logs'
down_revision: Union[str, Sequence[str], None] = '20250811_add_logged_date_columns'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'weight_logs',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('weight_kg', sa.Float(), nullable=False),
        sa.Column('logged_at', sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column('logged_date', sa.Date(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_weight_logs_logged_date', 'weight_logs', ['logged_date'])


def downgrade() -> None:
    op.drop_index('ix_weight_logs_logged_date', table_name='weight_logs')
    op.drop_table('weight_logs')


