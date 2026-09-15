"""Persist provider and prompt provenance."""

import sqlalchemy as sa

from alembic import op

revision = "20260911_03"
down_revision = "20260910_02"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("analyses", sa.Column("pipeline_metadata_json", sa.Text(), nullable=True))


def downgrade():
    op.drop_column("analyses", "pipeline_metadata_json")
