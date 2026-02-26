"""Loader: pushes transformed DataFrames into PostgreSQL.

Uses SQLAlchemy for DataFrame-to-table loading and raw SQL execution.
Each load is a TRUNCATE + INSERT (full refresh), making the pipeline idempotent —
you can re-run it any number of times and get the same result.
"""

from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from backend.etl.config import ETLConfig


class Loader:
    """Loads transformed DataFrames into PostgreSQL tables."""

    def __init__(self, config: ETLConfig):
        self._config = config
        self._engine: Engine = create_engine(config.database_url)

    @property
    def engine(self) -> Engine:
        """Expose engine for other modules that need DB access."""
        return self._engine

    def execute_sql_file(self, filepath: str | Path) -> None:
        """Execute a SQL file against the database.

        Used to run schema.sql for table creation and any standalone
        SQL scripts (indexes, views, etc.).

        Args:
            filepath: Path to .sql file.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"SQL file not found: {filepath}")

        sql_content = filepath.read_text()

        with self._engine.begin() as conn:
            # Split on semicolons to handle multi-statement files
            statements = [s.strip() for s in sql_content.split(";") if s.strip()]
            for stmt in statements:
                conn.execute(text(stmt))

        print(f"  Executed SQL file: {filepath.name}")

    def load_table(
        self,
        df: pd.DataFrame,
        table_name: str,
        if_exists: str = "append",
    ) -> int:
        """Load a DataFrame into a PostgreSQL table.

        Default behavior is TRUNCATE then APPEND, which gives us idempotent
        full-refresh semantics while preserving the table structure (constraints,
        indexes) that 'replace' would destroy.

        Args:
            df: DataFrame to load. Column names must match table columns.
            table_name: Target table name in PostgreSQL.
            if_exists: "append" (default), "replace", or "fail".

        Returns:
            Number of rows loaded.
        """
        # Truncate first for idempotent full refresh
        with self._engine.begin() as conn:
            conn.execute(text(f"TRUNCATE TABLE {table_name} CASCADE"))

        df.to_sql(
            name=table_name,
            con=self._engine,
            if_exists="append",
            index=False,
            chunksize=1000,
        )

        row_count = len(df)
        print(f"  Loaded {row_count:,} rows → {table_name}")
        return row_count

    def verify_row_count(self, table_name: str) -> int:
        """Query actual row count from a table for validation.

        Args:
            table_name: Table to count.

        Returns:
            Row count from database.
        """
        with self._engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            return result.scalar()

    def verify_referential_integrity(self) -> list[dict]:
        """Check that all FKs in fact_transactions resolve to dimension tables.

        Returns:
            List of check results: [{check, passed, orphan_count}]
        """
        checks = [
            {
                "check": "fact → dim_customers",
                "query": """
                    SELECT COUNT(*) FROM fact_transactions f
                    LEFT JOIN dim_customers c ON f.customer_id = c.customer_id
                    WHERE c.customer_id IS NULL
                """,
            },
            {
                "check": "fact → dim_products",
                "query": """
                    SELECT COUNT(*) FROM fact_transactions f
                    LEFT JOIN dim_products p ON f.product_id = p.product_id
                    WHERE p.product_id IS NULL
                """,
            },
            {
                "check": "fact → dim_dates",
                "query": """
                    SELECT COUNT(*) FROM fact_transactions f
                    LEFT JOIN dim_dates d ON f.date_id = d.date_id
                    WHERE d.date_id IS NULL
                """,
            },
        ]

        results = []
        with self._engine.connect() as conn:
            for check in checks:
                orphans = conn.execute(text(check["query"])).scalar()
                results.append({
                    "check": check["check"],
                    "passed": orphans == 0,
                    "orphan_count": orphans,
                })

        return results