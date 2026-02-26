"""Transformer: reshapes cleaned flat data into star schema tables.

Each build_dim_* method produces one dimension table. build_fact_transactions
produces the fact table with foreign keys to all dimensions.

The key principle: dimensions are built first (they define the ID mappings),
then the fact table references those IDs. This order is enforced by Pipeline.
"""

import pandas as pd
import numpy as np

from backend.etl.config import ETLConfig


# ── Region mapping (Country → Region) ─────────────────────────
# Used for the synthetic region field on dim_customers.
# Groups the ~40 countries into meaningful regions for dashboard filtering.

COUNTRY_TO_REGION = {
    "United Kingdom": "UK",
    "EIRE": "Europe",
    "France": "Europe",
    "Germany": "Europe",
    "Spain": "Europe",
    "Netherlands": "Europe",
    "Belgium": "Europe",
    "Switzerland": "Europe",
    "Portugal": "Europe",
    "Italy": "Europe",
    "Austria": "Europe",
    "Denmark": "Europe",
    "Norway": "Europe",
    "Sweden": "Europe",
    "Finland": "Europe",
    "Poland": "Europe",
    "Czech Republic": "Europe",
    "Greece": "Europe",
    "Iceland": "Europe",
    "Lithuania": "Europe",
    "Malta": "Europe",
    "Cyprus": "Europe",
    "Channel Islands": "Europe",
    "European Community": "Europe",
    "USA": "North America",
    "Canada": "North America",
    "Brazil": "South America",
    "RSA": "Africa",
    "Nigeria": "Africa",
    "Australia": "Asia Pacific",
    "Japan": "Asia Pacific",
    "Singapore": "Asia Pacific",
    "Hong Kong": "Asia Pacific",
    "Korea": "Asia Pacific",
    "Thailand": "Asia Pacific",
    "Israel": "Middle East",
    "Lebanon": "Middle East",
    "Bahrain": "Middle East",
    "Saudi Arabia": "Middle East",
    "United Arab Emirates": "Middle East",
    "Unspecified": "Unknown",
}

# ── Acquisition channel options for synthetic generation ───────

ACQUISITION_CHANNELS = [
    "Organic Search",
    "Direct",
    "Referral",
    "Paid Search",
    "Email",
    "Social Media",
]

# Weighted probabilities (realistic distribution)
CHANNEL_WEIGHTS = [0.30, 0.25, 0.20, 0.12, 0.08, 0.05]

# ── Age group options for synthetic generation ─────────────────

AGE_GROUPS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
AGE_WEIGHTS = [0.08, 0.25, 0.28, 0.22, 0.12, 0.05]


class Transformer:
    """Transforms cleaned data into star schema dimension and fact tables."""

    def __init__(self, config: ETLConfig):
        self._config = config
        self._rng = np.random.default_rng(42)  # Reproducible synthetic data

    def build_dim_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build date dimension from the full date range in the data.

        Creates one row per calendar date (not just dates with transactions),
        covering the entire range. This allows the dashboard to show gaps
        (days with zero revenue) rather than skipping them.

        Args:
            df: Cleaned transaction DataFrame (needs 'invoice_date' column).

        Returns:
            DataFrame with columns matching dim_dates schema.
        """
        min_date = df["invoice_date"].dt.date.min()
        max_date = df["invoice_date"].dt.date.max()

        date_range = pd.date_range(start=min_date, end=max_date, freq="D")

        dim_dates = pd.DataFrame({"full_date": date_range.date})
        dim_dates["date_id"] = range(1, len(dim_dates) + 1)

        dt = pd.to_datetime(dim_dates["full_date"])
        dim_dates["day_of_week"] = dt.dt.dayofweek           # 0=Mon, 6=Sun
        dim_dates["day_name"] = dt.dt.day_name()
        dim_dates["month"] = dt.dt.month
        dim_dates["month_name"] = dt.dt.month_name()
        dim_dates["quarter"] = dt.dt.quarter
        dim_dates["year"] = dt.dt.year
        dim_dates["is_weekend"] = dim_dates["day_of_week"].isin([5, 6])
        dim_dates["is_holiday"] = False  # Placeholder — can enhance with UK holiday calendar

        return dim_dates[["date_id", "full_date", "day_of_week", "day_name",
                          "month", "month_name", "quarter", "year",
                          "is_holiday", "is_weekend"]]

    def build_dim_products(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build product dimension from unique stock codes.

        One row per unique stock_code. Description takes the most common
        description for that code (handles minor description variations).
        avg_unit_price is computed from all non-return transactions.

        Args:
            df: Cleaned transaction DataFrame.

        Returns:
            DataFrame with columns matching dim_products schema.
        """
        non_returns = df[~df["is_return"]]

        # Most common description per stock_code
        descriptions = (
            non_returns.groupby("stock_code")["description"]
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
        )

        # Average price per stock_code
        avg_prices = non_returns.groupby("stock_code")["price"].mean().round(2)

        dim_products = pd.DataFrame({
            "stock_code": descriptions.index,
            "description": descriptions.values,
            "avg_unit_price": avg_prices.reindex(descriptions.index).values,
        })

        dim_products["product_id"] = range(1, len(dim_products) + 1)
        dim_products["category"] = None  # Populated later via description clustering

        return dim_products[["product_id", "stock_code", "description",
                             "category", "avg_unit_price"]]

    def build_dim_customers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build customer dimension with synthetic demographic fields.

        Only includes rows where customer_id is not null. Computes first/last
        purchase dates from transaction data, then generates synthetic
        age_group, region, and acquisition_channel.

        WHY synthetic fields: The real dataset has no demographics. Adding them
        simulates the real-world scenario of joining multiple data sources,
        and gives the dashboard more interesting filter/segment dimensions.

        Args:
            df: Cleaned transaction DataFrame (will filter to non-null customer_id).

        Returns:
            DataFrame with columns matching dim_customers schema.
        """
        cust_df = df[df["customer_id"].notna()].copy()

        # Aggregate per customer
        customer_agg = cust_df.groupby("customer_id").agg(
            country=("country", "first"),          # Take first seen country
            first_purchase=("invoice_date", "min"),
            last_purchase=("invoice_date", "max"),
        ).reset_index()

        customer_agg["customer_id"] = customer_agg["customer_id"].astype(int)
        customer_agg["first_purchase"] = customer_agg["first_purchase"].dt.date
        customer_agg["last_purchase"] = customer_agg["last_purchase"].dt.date

        n = len(customer_agg)

        # Synthetic: region (derived from country, not random)
        customer_agg["region"] = (
            customer_agg["country"]
            .map(COUNTRY_TO_REGION)
            .fillna("Other")
        )

        # Synthetic: acquisition channel (random but seeded)
        customer_agg["acquisition_channel"] = self._rng.choice(
            ACQUISITION_CHANNELS, size=n, p=CHANNEL_WEIGHTS
        )

        # Synthetic: age group (random but seeded)
        customer_agg["age_group"] = self._rng.choice(
            AGE_GROUPS, size=n, p=AGE_WEIGHTS
        )

        return customer_agg[["customer_id", "country", "first_purchase",
                             "last_purchase", "age_group", "region",
                             "acquisition_channel"]]

    def build_fact_transactions(
        self,
        df: pd.DataFrame,
        dim_customers: pd.DataFrame,
        dim_products: pd.DataFrame,
        dim_dates: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build fact table by joining cleaned data to dimension keys.

        Only includes rows with valid customer_id (non-null) since the fact
        table has a NOT NULL FK constraint to dim_customers.

        Args:
            df: Cleaned transaction DataFrame.
            dim_customers: Built customer dimension (for ID validation).
            dim_products: Built product dimension (for stock_code → product_id).
            dim_dates: Built date dimension (for date → date_id).

        Returns:
            DataFrame with columns matching fact_transactions schema.
        """
        # Filter to rows with customer_id (fact table requires it)
        fact_df = df[df["customer_id"].notna()].copy()
        fact_df["customer_id"] = fact_df["customer_id"].astype(int)

        # Map stock_code → product_id
        product_map = dim_products.set_index("stock_code")["product_id"]
        fact_df["product_id"] = fact_df["stock_code"].map(product_map)

        # Map invoice_date → date_id
        date_map = dim_dates.set_index("full_date")["date_id"]
        fact_df["txn_date"] = fact_df["invoice_date"].dt.date
        fact_df["date_id"] = fact_df["txn_date"].map(date_map)

        # Drop rows that didn't match a dimension (shouldn't happen, but safety)
        before = len(fact_df)
        fact_df = fact_df.dropna(subset=["product_id", "date_id"])
        fact_df["product_id"] = fact_df["product_id"].astype(int)
        fact_df["date_id"] = fact_df["date_id"].astype(int)
        dropped = before - len(fact_df)
        if dropped > 0:
            print(f"    ⚠ Dropped {dropped:,} rows with no dimension match")

        # Filter to only customers that exist in dim_customers
        valid_customers = set(dim_customers["customer_id"])
        fact_df = fact_df[fact_df["customer_id"].isin(valid_customers)]

        # Select and rename to match schema
        fact_df = fact_df.rename(columns={
            "invoice": "invoice_no",
            "price": "unit_price",
        })

        fact_df = fact_df[["invoice_no", "customer_id", "product_id", "date_id",
                           "quantity", "unit_price", "total_amount", "is_return"]]

        fact_df = fact_df.reset_index(drop=True)
        fact_df.index += 1
        fact_df.index.name = "transaction_id"

        return fact_df.reset_index()