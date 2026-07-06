"""knowledge_items table + research_notes source/confidence/entities

Revision ID: b7e2f1a90c44
Revises: adcc4eb838ca
Create Date: 2026-07-05

"""
from alembic import op
import sqlalchemy as sa


revision = 'b7e2f1a90c44'
down_revision = 'adcc4eb838ca'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'knowledge_items',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('knowledge_type', sa.String(length=30), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('tickers', sa.JSON(), nullable=True),
        sa.Column('regime', sa.String(length=30), nullable=True),
        sa.Column('confidence', sa.Numeric(precision=4, scale=3), nullable=False),
        sa.Column('source', sa.String(length=60), nullable=True),
        sa.Column('conditions', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_knowledge_items_knowledge_type'),
                    'knowledge_items', ['knowledge_type'])
    op.create_index(op.f('ix_knowledge_items_regime'),
                    'knowledge_items', ['regime'])

    with op.batch_alter_table('research_notes') as batch:
        batch.add_column(sa.Column('source', sa.String(length=40), nullable=True))
        batch.add_column(sa.Column('confidence',
                                   sa.Numeric(precision=4, scale=3), nullable=True))
        batch.add_column(sa.Column('entities', sa.JSON(), nullable=True))
    op.create_index(op.f('ix_research_notes_source'), 'research_notes', ['source'])


def downgrade() -> None:
    op.drop_index(op.f('ix_research_notes_source'), table_name='research_notes')
    with op.batch_alter_table('research_notes') as batch:
        batch.drop_column('entities')
        batch.drop_column('confidence')
        batch.drop_column('source')
    op.drop_index(op.f('ix_knowledge_items_regime'), table_name='knowledge_items')
    op.drop_index(op.f('ix_knowledge_items_knowledge_type'),
                  table_name='knowledge_items')
    op.drop_table('knowledge_items')
