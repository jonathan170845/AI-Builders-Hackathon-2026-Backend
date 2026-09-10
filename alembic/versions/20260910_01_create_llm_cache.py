"""create llm cache

Revision ID: 20260910_01
Revises:
Create Date: 2026-09-10 00:00:00
"""

import sqlalchemy as sa

from alembic import op

revision = "20260910_01"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "llm_cache",
        sa.Column("cache_key", sa.String(length=64), nullable=False),
        sa.Column("model", sa.String(length=255), nullable=False),
        sa.Column("prompt_version", sa.String(length=100), nullable=False),
        sa.Column("response_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_accessed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("response_bytes", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("cache_key"),
    )
    op.create_index("ix_llm_cache_expires_at", "llm_cache", ["expires_at"], unique=False)
    op.create_index(
        "ix_llm_cache_last_accessed_at", "llm_cache", ["last_accessed_at"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_llm_cache_last_accessed_at", table_name="llm_cache")
    op.drop_index("ix_llm_cache_expires_at", table_name="llm_cache")
    op.drop_table("llm_cache")
