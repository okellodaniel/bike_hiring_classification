"""added humidity column

Revision ID: 2833e470427f
Revises: 70971ffd486e
Create Date: 2025-01-31 06:33:19.052597

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '2833e470427f'
down_revision = '70971ffd486e'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('bikeshares', schema=None) as batch_op:
        batch_op.add_column(sa.Column('humidity', sa.Float(), nullable=False))

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('bikeshares', schema=None) as batch_op:
        batch_op.drop_column('humidity')

    # ### end Alembic commands ###
