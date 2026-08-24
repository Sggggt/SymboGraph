"""Make the repository-root .env the sole Runtime Settings authority.

Revision ID: 20260822_0043
Revises: 20260820_0041

The migration preserves only non-secret audit metadata.  Historical settings
snapshots and secret-presence mirrors are intentionally not copied.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260822_0043"
down_revision = "20260820_0041"
branch_labels = None
depends_on = None


def _tables() -> set[str]:
    return set(sa.inspect(op.get_bind()).get_table_names())


def upgrade() -> None:
    tables = _tables()
    if "runtime_settings_audits" not in tables:
        op.create_table(
            "runtime_settings_audits",
            sa.Column("id", sa.String(length=36), primary_key=True),
            sa.Column("protocol_version", sa.String(length=64), nullable=False),
            sa.Column("version_hash", sa.String(length=64), nullable=False),
            sa.Column("prior_runtime_version_hash", sa.String(length=64), nullable=True),
            sa.Column("changed_keys_json", sa.JSON(), nullable=False),
            sa.Column("lifecycle_json", sa.JSON(), nullable=False),
            sa.Column("field_status_json", sa.JSON(), nullable=False),
            sa.Column("status", sa.String(length=32), nullable=False),
            sa.Column("env_identity_hash", sa.String(length=64), nullable=True),
            sa.Column("runtime_version_hash", sa.String(length=64), nullable=True),
            sa.Column("source", sa.String(length=64), nullable=False),
            sa.Column("last_error_type", sa.String(length=128), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint("version_hash", name="uq_runtime_settings_audits_hash"),
            sa.CheckConstraint(
                "protocol_version = 'runtime_settings_audit_v1'",
                name="ck_runtime_settings_audits_protocol",
            ),
            sa.CheckConstraint(
                "status IN ('written','applied','pending_lifecycle','failed')",
                name="ck_runtime_settings_audits_status",
            ),
        )
        op.create_index(
            "ix_runtime_settings_audits_status_created",
            "runtime_settings_audits",
            ["status", "created_at"],
        )
        op.create_index("ix_rs_audit_prior_runtime", "runtime_settings_audits", ["prior_runtime_version_hash"])
        op.create_index("ix_rs_audit_env_identity", "runtime_settings_audits", ["env_identity_hash"])
        op.create_index("ix_rs_audit_runtime", "runtime_settings_audits", ["runtime_version_hash"])
        op.create_index("ix_runtime_settings_audits_version_hash", "runtime_settings_audits", ["version_hash"])
        op.create_index("ix_runtime_settings_audits_source", "runtime_settings_audits", ["source"])
        op.create_index("ix_runtime_settings_audits_created_at", "runtime_settings_audits", ["created_at"])

    tables = _tables()
    if "runtime_settings_desired_versions" in tables:
        op.execute(
            sa.text(
                """
                INSERT INTO runtime_settings_audits (
                    id, protocol_version, version_hash, prior_runtime_version_hash,
                    changed_keys_json, lifecycle_json, field_status_json, status,
                    env_identity_hash, runtime_version_hash, source, last_error_type,
                    created_at, updated_at
                )
                SELECT id, 'runtime_settings_audit_v1', version_hash,
                    base_active_version_hash, changed_keys_json, lifecycle_json,
                    '{}'::json, 'written', desired_env_identity_hash,
                    active_runtime_version_hash, source, last_error_type,
                    created_at, updated_at
                FROM runtime_settings_desired_versions
                ON CONFLICT (version_hash) DO NOTHING
                """
            )
        )
        op.drop_table("runtime_settings_desired_versions")

    columns = {
        column["name"]
        for column in sa.inspect(op.get_bind()).get_columns("runtime_settings_versions")
    }
    if "settings_json" in columns:
        op.drop_column("runtime_settings_versions", "settings_json")


def downgrade() -> None:
    raise RuntimeError(
        "The single-root-env migration deliberately removes competing settings "
        "snapshots; restore a database backup instead of recreating them."
    )
