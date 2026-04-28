"""Postgres feature-store utilities for forecaster v1."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    sql_type: str
    nullable: bool = True


CURATED_TABLE = "hdb_resale_forecaster_v1_curated"
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

CURATED_SCHEMA: tuple[ColumnSpec, ...] = (
    ColumnSpec("month", "TIMESTAMP", False),
    ColumnSpec("town", "TEXT", False),
    ColumnSpec("flat_type", "TEXT", False),
    ColumnSpec("flat_model", "TEXT", False),
    ColumnSpec("storey_range", "TEXT", True),
    ColumnSpec("floor_area_sqm", "DOUBLE PRECISION", False),
    ColumnSpec("lease_commence_date", "DOUBLE PRECISION", True),
    ColumnSpec("remaining_lease_years", "DOUBLE PRECISION", False),
    ColumnSpec("resale_price", "DOUBLE PRECISION", False),
    ColumnSpec("year", "INTEGER", False),
    ColumnSpec("month_num", "INTEGER", False),
    ColumnSpec("source_path", "TEXT", False),
    ColumnSpec("source_sha256", "TEXT", False),
    ColumnSpec("ingested_at_utc", "TEXT", False),
)


def validate_schema(df: pd.DataFrame, schema: Iterable[ColumnSpec] = CURATED_SCHEMA) -> None:
    missing = [c.name for c in schema if c.name not in df.columns]
    if missing:
        raise KeyError(f"Missing curated schema columns: {missing}")
    for col in schema:
        if not col.nullable and df[col.name].isna().any():
            raise ValueError(f"Non-nullable column has nulls: {col.name}")


def validate_identifier(name: str, *, label: str) -> str:
    candidate = str(name).strip()
    if not candidate:
        raise ValueError(f"{label} must be non-empty.")
    if not _IDENTIFIER_RE.fullmatch(candidate):
        raise ValueError(
            f"{label} must match {_IDENTIFIER_RE.pattern} (letters, digits, underscore)."
        )
    return candidate


def write_curated_to_postgres(
    df: pd.DataFrame,
    *,
    dsn: str,
    table_name: str = CURATED_TABLE,
    schema_name: str = "public",
    if_exists: str = "replace",
) -> int:
    """Write curated frame to Postgres table with schema contract checks."""
    validate_schema(df)
    try:
        import psycopg
        from psycopg import sql
    except ImportError as ex:
        raise RuntimeError("psycopg is required for Postgres writes.") from ex

    if if_exists not in {"replace", "append"}:
        raise ValueError("if_exists must be 'replace' or 'append'")

    schema_name = validate_identifier(schema_name, label="schema_name")
    table_name = validate_identifier(table_name, label="table_name")
    col_names = [c.name for c in CURATED_SCHEMA]
    for col in col_names:
        validate_identifier(col, label=f"column name '{col}'")
    col_defs = [
        sql.SQL("{} {}{}").format(
            sql.Identifier(c.name),
            sql.SQL(c.sql_type),
            sql.SQL("" if c.nullable else " NOT NULL"),
        )
        for c in CURATED_SCHEMA
    ]
    fqtn = sql.SQL("{}.{}").format(sql.Identifier(schema_name), sql.Identifier(table_name))
    columns_sql = sql.SQL(", ").join(col_defs)
    col_list = sql.SQL(", ").join(sql.Identifier(c) for c in col_names)
    placeholders = sql.SQL(", ").join(sql.Placeholder() for _ in col_names)

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
                    sql.Identifier(schema_name)
                )
            )
            if if_exists == "replace":
                cur.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(fqtn))
                cur.execute(sql.SQL("CREATE TABLE {} ({})").format(fqtn, columns_sql))
            elif if_exists == "append":
                cur.execute(sql.SQL("CREATE TABLE IF NOT EXISTS {} ({})").format(fqtn, columns_sql))

            insert_sql = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
                fqtn, col_list, placeholders
            )
            values = [tuple(row[c] for c in col_names) for row in df.to_dict(orient="records")]
            cur.executemany(insert_sql, values)
        conn.commit()

    return int(len(df))
