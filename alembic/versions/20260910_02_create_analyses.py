"""create analyses

Revision ID: 20260910_02
Revises: 20260910_01
Create Date: 2026-09-10 00:00:00
"""

import sqlalchemy as sa

from alembic import op

revision = "20260910_02"
down_revision = "20260910_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    status_enum = sa.Enum(
        "queued",
        "processing",
        "completed",
        "failed",
        name="analysis_status",
        native_enum=False,
        create_constraint=True,
    )
    stage_enum = sa.Enum(
        "queued",
        "extracting_assumptions",
        "retrieving_evidence",
        "calculating_financials",
        "reviewing_evidence",
        "building_report",
        name="analysis_stage",
        native_enum=False,
        create_constraint=True,
    )
    op.create_table(
        "analyses",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("decision", sa.Text(), nullable=False),
        sa.Column("financial_inputs_json", sa.Text(), nullable=True),
        sa.Column("status", status_enum, nullable=False),
        sa.Column("stage", stage_enum, nullable=True),
        sa.Column("result_json", sa.Text(), nullable=True),
        sa.Column("public_error_code", sa.String(length=100), nullable=True),
        sa.Column("public_error_message", sa.Text(), nullable=True),
        sa.Column("internal_error", sa.Text(), nullable=True),
        sa.Column("pipeline_version", sa.String(length=100), nullable=False),
        sa.Column("assumption_count", sa.Integer(), nullable=False),
        sa.Column("financial_warning_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_analyses_created_at_id", "analyses", ["created_at", "id"], unique=False)
    op.create_index("ix_analyses_status", "analyses", ["status"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_analyses_status", table_name="analyses")
    op.drop_index("ix_analyses_created_at_id", table_name="analyses")
    op.drop_table("analyses")
