from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '20250810_add_daily_summaries'
down_revision = '20250809_add_sport_columns'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'daily_summaries',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('calories_total', sa.Float(), nullable=False, server_default='0'),
        sa.Column('protein_g_total', sa.Float(), nullable=False, server_default='0'),
        sa.Column('carbs_g_total', sa.Float(), nullable=False, server_default='0'),
        sa.Column('fat_g_total', sa.Float(), nullable=False, server_default='0'),
        sa.Column('sport_calories_total', sa.Float(), nullable=False, server_default='0'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id', 'date', name='uq_daily_summaries_user_date')
    )
    op.create_index(op.f('ix_daily_summaries_date'), 'daily_summaries', ['date'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_daily_summaries_date'), table_name='daily_summaries')
    op.drop_table('daily_summaries')


