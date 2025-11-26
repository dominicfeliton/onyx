"""Add per-persona search prompts

Revision ID: 6f9a23b4c5d6
Revises: 4d7c04e9980e
Create Date: 2025-11-26 00:00:00.000000

"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "6f9a23b4c5d6"
down_revision = "4d7c04e9980e"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add per-persona search prompt columns
    op.add_column(
        "persona",
        sa.Column("search_tool_description", sa.Text(), nullable=True),
    )
    op.add_column(
        "persona",
        sa.Column("history_rephrase_prompt", sa.Text(), nullable=True),
    )
    op.add_column(
        "persona",
        sa.Column("search_decision_prompt", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("persona", "search_decision_prompt")
    op.drop_column("persona", "history_rephrase_prompt")
    op.drop_column("persona", "search_tool_description")
