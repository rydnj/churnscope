"""Pipeline: orchestrates the full ETL flow.

This is the only file you run. It calls Extract → Clean → Transform → Load
in the correct order, validates the results, and produces a summary report.

Usage:
    python -m backend.etl.pipeline
"""

import time
from dataclasses import dataclass, field

from backend.etl.config import ETLConfig, SQL_DIR
from backend.etl.extract import Extractor
from backend.etl.clean import Cleaner
from backend.etl.transform import Transformer
from backend.etl.load import Loader


@dataclass
class PipelineResult:
    """Summary of a pipeline run."""
    success: bool
    rows_loaded: dict[str, int] = field(default_factory=dict)
    cleaning_report: dict = field(default_factory=dict)
    validation: list[dict] = field(default_factory=list)
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


class Pipeline:
    """Orchestrates the full ETL flow. Idempotent — safe to re-run."""

    def __init__(self, config: ETLConfig | None = None):
        self._config = config or ETLConfig()
        self._extractor = Extractor(self._config)
        self._cleaner = Cleaner(self._config)
        self._transformer = Transformer(self._config)
        self._loader = Loader(self._config)

    def run(self) -> PipelineResult:
        """Execute the full pipeline: Extract → Clean → Transform → Load → Validate.

        Returns:
            PipelineResult with row counts, cleaning stats, and validation results.
        """
        result = PipelineResult(success=False)
        start = time.time()

        try:
            # ── Step 1: Create/reset schema ───────────────────────
            print("\n[1/6] Creating database schema...")
            self._loader.execute_sql_file(SQL_DIR / "schema.sql")

            # ── Step 2: Extract ───────────────────────────────────
            print("\n[2/6] Extracting raw data...")
            raw_df = self._extractor.extract_transactions()

            # ── Step 3: Clean ─────────────────────────────────────
            print("\n[3/6] Cleaning data...")
            clean_df = self._cleaner.clean_transactions(raw_df)
            result.cleaning_report = self._cleaner.get_cleaning_report()

            # ── Step 4: Transform into star schema ────────────────
            print("\n[4/6] Transforming to star schema...")
            print("  Building dim_dates...")
            dim_dates = self._transformer.build_dim_dates(clean_df)
            print(f"    → {len(dim_dates):,} date rows")

            print("  Building dim_products...")
            dim_products = self._transformer.build_dim_products(clean_df)
            print(f"    → {len(dim_products):,} products")

            print("  Building dim_customers...")
            dim_customers = self._transformer.build_dim_customers(clean_df)
            print(f"    → {len(dim_customers):,} customers")

            print("  Building fact_transactions...")
            fact_txn = self._transformer.build_fact_transactions(
                clean_df, dim_customers, dim_products, dim_dates
            )
            print(f"    → {len(fact_txn):,} fact rows")

            # ── Step 5: Load ──────────────────────────────────────
            # Order matters: dimensions before fact (FK constraints)
            print("\n[5/6] Loading to PostgreSQL...")
            result.rows_loaded["dim_dates"] = self._loader.load_table(dim_dates, "dim_dates")
            result.rows_loaded["dim_products"] = self._loader.load_table(dim_products, "dim_products")
            result.rows_loaded["dim_customers"] = self._loader.load_table(dim_customers, "dim_customers")
            result.rows_loaded["fact_transactions"] = self._loader.load_table(fact_txn, "fact_transactions")

            # ── Step 6: Validate ──────────────────────────────────
            print("\n[6/6] Validating...")
            result.validation = self._validate(result.rows_loaded)

            all_passed = all(v["passed"] for v in result.validation)
            result.success = all_passed

            if all_passed:
                print("\n✅ Pipeline completed successfully!")
            else:
                failed = [v for v in result.validation if not v["passed"]]
                print(f"\n⚠️  Pipeline completed with {len(failed)} validation warning(s)")
                for f in failed:
                    print(f"   FAILED: {f['check']} — {f.get('details', '')}")

        except Exception as e:
            result.errors.append(str(e))
            print(f"\n❌ Pipeline failed: {e}")
            raise

        finally:
            result.duration_seconds = round(time.time() - start, 2)
            print(f"\nDuration: {result.duration_seconds}s")

        return result

    def _validate(self, rows_loaded: dict[str, int]) -> list[dict]:
        """Run all post-load validation checks."""
        checks = []

        # Check 1: Row counts match what we loaded
        for table_name, expected in rows_loaded.items():
            actual = self._loader.verify_row_count(table_name)
            checks.append({
                "check": f"row_count:{table_name}",
                "passed": actual == expected,
                "details": f"expected={expected:,}, actual={actual:,}",
            })

        # Check 2: Referential integrity (fact → dimensions)
        fk_checks = self._loader.verify_referential_integrity()
        for fk in fk_checks:
            checks.append({
                "check": f"fk_integrity:{fk['check']}",
                "passed": fk["passed"],
                "details": f"orphan_rows={fk['orphan_count']}",
            })

        # Check 3: No null customer_ids in fact table
        with self._loader.engine.connect() as conn:
            from sqlalchemy import text
            null_customers = conn.execute(
                text("SELECT COUNT(*) FROM fact_transactions WHERE customer_id IS NULL")
            ).scalar()
            checks.append({
                "check": "no_null_customer_id_in_fact",
                "passed": null_customers == 0,
                "details": f"null_customer_rows={null_customers}",
            })

        for c in checks:
            status = "✓" if c["passed"] else "✗"
            print(f"  {status} {c['check']}: {c['details']}")

        return checks


# ── CLI entry point ────────────────────────────────────────────

def main():
    """Run the full pipeline from command line."""
    print("=" * 60)
    print("ChurnScope ETL Pipeline")
    print("=" * 60)

    config = ETLConfig()
    pipeline = Pipeline(config)
    result = pipeline.run()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Success: {result.success}")
    print(f"Duration: {result.duration_seconds}s")
    print(f"Rows loaded:")
    for table, count in result.rows_loaded.items():
        print(f"  {table}: {count:,}")
    print(f"Cleaning: {result.cleaning_report.get('initial_rows', '?'):,} → "
          f"{result.cleaning_report.get('final_rows', '?'):,} "
          f"({result.cleaning_report.get('pct_dropped', '?')}% dropped)")

    return result


if __name__ == "__main__":
    main()