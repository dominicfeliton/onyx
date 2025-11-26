"""Add use_non_tool_calling_fast to model_configuration

Revision ID: 4d7c04e9980e
Revises: c7e9f4a3b2d1
Create Date: 2025-11-25 00:00:00.000000

"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "4d7c04e9980e"
down_revision = "c7e9f4a3b2d1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "model_configuration",
        sa.Column("use_non_tool_calling_fast", sa.Boolean(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("model_configuration", "use_non_tool_calling_fast")
