"""Cleaner: all data quality operations in a defined, auditable sequence.

Each step is documented with WHY it exists, traced back to Phase 0 findings.
The cleaning report captures before/after counts so you can verify and explain
exactly what happened to the data — critical for interviews and debugging.
"""

import pandas as pd
import numpy as np

from backend.etl.config import ETLConfig, RAW_COLUMNS


class Cleaner:
    """Handles all data quality operations on raw transaction data."""

    def __init__(self, config: ETLConfig):
        self._config = config
        self._report: dict = {}

    def clean_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all cleaning steps in order. Returns cleaned DataFrame.

        The output DataFrame has standardized column names (snake_case),
        computed fields (is_return, total_amount), and all quality issues resolved.

        Args:
            df: Raw combined DataFrame from Extractor.

        Returns:
            Cleaned DataFrame ready for Transformer.
        """
        self._report = {"steps": []}
        initial_rows = len(df)

        # ── Step 1: Standardize column names ──────────────────────
        # WHY: Raw columns have spaces and mixed case ("Customer ID").
        # Snake_case is our convention (Section 2 of groundwork doc).
        df = self._standardize_columns(df)

        # ── Step 2: Strip whitespace from string columns ──────────
        # WHY: Phase 0 found leading/trailing spaces in Description,
        # e.g., " WHITE CHERRY LIGHTS", "PINK DOUGHNUT TRINKET POT ".
        df = self._strip_strings(df)

        # ── Step 3: Drop exact duplicate rows ─────────────────────
        # WHY: Dec 2010 exists in both Excel sheets.
        # Dedup on all columns to remove sheet overlap.
        before = len(df)
        df = df.drop_duplicates()
        self._log_step("drop_duplicates", before, len(df), "Sheet overlap + exact dupes")

        # ── Step 4: Flag returns ──────────────────────────────────
        # WHY: Cancellation invoices start with "C" (Phase 0: ~19,494 total).
        # We flag rather than drop — returns are valid for return_rate features.
        df["is_return"] = df["invoice"].astype(str).str.startswith("C")
        returns_count = df["is_return"].sum()
        self._report["returns_flagged"] = int(returns_count)

        # ── Step 5: Filter non-product stock codes ────────────────
        # WHY: Codes like POST, DOT, BANK CHARGES are service fees,
        # not product sales. They'd pollute product analysis and RFM.
        before = len(df)
        non_product_mask = (
            df["stock_code"].isin(self._config.non_product_stock_codes)
            | (df["stock_code"].str.len() == 1) & (df["stock_code"].str.isalpha())
        )
        df = df[~non_product_mask]
        self._log_step("filter_non_products", before, len(df), "POST, DOT, BANK CHARGES, etc.")

        # ── Step 6: Filter invalid prices ─────────────────────────
        # WHY: Phase 0 found prices down to -53,594. Negative prices
        # on non-return rows are data errors (adjustments, write-offs).
        # We keep negative prices on returns (they're legitimate reversals).
        before = len(df)
        invalid_price_mask = (
            (df["price"] <= self._config.min_valid_price)
            & (~df["is_return"])
        )
        df = df[~invalid_price_mask]
        self._log_step("filter_invalid_prices", before, len(df), "Price <= 0 on non-returns")

        # ── Step 7: Filter zero-quantity rows ─────────────────────
        # WHY: Quantity of 0 means nothing happened — no product moved.
        before = len(df)
        df = df[df["quantity"] != 0]
        self._log_step("filter_zero_quantity", before, len(df), "Quantity == 0")

        # ── Step 8: Cap quantity outliers ─────────────────────────
        # WHY: Phase 0 found range -80,995 to +80,995 with std ~200.
        # Extreme values distort RFM monetary scores and revenue calculations.
        # We cap at percentiles rather than drop — preserves the transaction.
        lower = df["quantity"].quantile(self._config.quantity_lower_percentile)
        upper = df["quantity"].quantile(self._config.quantity_upper_percentile)
        capped = df["quantity"].clip(lower=lower, upper=upper)
        rows_capped = (df["quantity"] != capped).sum()
        df["quantity"] = capped
        self._report["quantity_capped"] = int(rows_capped)
        self._report["quantity_cap_range"] = (int(lower), int(upper))

        # ── Step 9: Compute total_amount ──────────────────────────
        # WHY: Quantity * Price gives us the transaction value.
        # Negative for returns (negative quantity * positive price) — correct
        # for revenue calculations where returns should subtract.
        df["total_amount"] = df["quantity"] * df["price"]

        # ── Step 10: Cast customer_id to nullable int ─────────────
        # WHY: Raw column is float64 because of NaN. We keep NaN rows
        # (needed for aggregate analysis) but cast non-null to int.
        df["customer_id"] = df["customer_id"].astype("Int64")  # pandas nullable int

        # ── Final report ──────────────────────────────────────────
        self._report["initial_rows"] = initial_rows
        self._report["final_rows"] = len(df)
        self._report["rows_with_customer_id"] = int(df["customer_id"].notna().sum())
        self._report["rows_without_customer_id"] = int(df["customer_id"].isna().sum())
        self._report["pct_dropped"] = round(
            (1 - len(df) / initial_rows) * 100, 2
        )

        print(f"  Cleaning complete: {initial_rows:,} → {len(df):,} rows "
              f"({self._report['pct_dropped']}% dropped)")
        return df

    def get_cleaning_report(self) -> dict:
        """Return detailed cleaning statistics.

        Call after clean_transactions(). Used for pipeline validation
        and for documenting data quality in the README.
        """
        return self._report

    # ── Private helpers ────────────────────────────────────────────

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename raw columns to snake_case convention."""
        rename_map = {v: k for k, v in RAW_COLUMNS.items()}
        return df.rename(columns=rename_map)

    def _strip_strings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Strip leading/trailing whitespace from all string columns.

        Also forces object-dtype columns to actual strings first,
        because pandas reads mixed-type columns (like Invoice, which
        has both ints and strings like 'C489434') as object dtype
        where .str.strip() on non-string values returns NaN.
        """
        str_cols = df.select_dtypes(include="object").columns
        for col in str_cols:
            df[col] = df[col].astype(str).str.strip()
            # Restore actual NaN for values that were originally NaN
            df.loc[df[col] == "nan", col] = None
        return df

    def _log_step(self, step_name: str, before: int, after: int, reason: str) -> None:
        """Record a cleaning step for the report."""
        removed = before - after
        self._report["steps"].append({
            "step": step_name,
            "rows_before": before,
            "rows_after": after,
            "rows_removed": removed,
            "reason": reason,
        })
        print(f"    {step_name}: removed {removed:,} rows ({reason})")