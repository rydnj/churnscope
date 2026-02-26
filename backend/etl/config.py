"""Central configuration for ChurnScope ETL pipeline.

Every constant, threshold, and path used across the ETL lives here.
No other module hardcodes these values — they import from this config.
"""

from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# ── Project paths ──────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # churnscope/
BACKEND_DIR = PROJECT_ROOT / "backend"
DATA_DIR = PROJECT_ROOT / "data"
SQL_DIR = BACKEND_DIR / "sql"

# ── Raw data ───────────────────────────────────────────────────

RAW_DATA_FILENAME = "online_retail_II.xlsx"
RAW_DATA_PATH = DATA_DIR / RAW_DATA_FILENAME
SHEET_NAMES = ["Year 2009-2010", "Year 2010-2011"]

# ── Database ───────────────────────────────────────────────────

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://churnscope:churnscope_dev@localhost:5433/churnscope")

# ── Raw column names (exact match to Excel file) ──────────────

RAW_COLUMNS = {
    "invoice": "Invoice",
    "stock_code": "StockCode",
    "description": "Description",
    "quantity": "Quantity",
    "invoice_date": "InvoiceDate",
    "price": "Price",
    "customer_id": "Customer ID",
    "country": "Country",
}

# ── Non-product stock codes to exclude ─────────────────────────
# These are service charges, adjustments, and fees — not real products.
# Verified against actual data during Phase 0 inspection.

NON_PRODUCT_STOCK_CODES = {
    "POST", "POSTAGE", "DOT", "M", "MANUAL",
    "BANK CHARGES", "PADS", "C2", "AMAZONFEE",
    "CRUK", "D", "S", "B",
}

# Also exclude stock codes that are purely single uppercase letters
# (caught by length check + alpha check in Cleaner)

# ── Cleaning thresholds ────────────────────────────────────────

MIN_VALID_PRICE = 0.001           # Exclude zero/negative prices on non-returns
QUANTITY_LOWER_PERCENTILE = 0.01  # Cap extreme negative quantities
QUANTITY_UPPER_PERCENTILE = 0.99  # Cap extreme positive quantities

# ── Business logic constants ───────────────────────────────────

CHURN_THRESHOLD_DAYS = 90         # No purchase in 90+ days = churned
RFM_QUANTILES = 5                 # Quintile scoring (1-5) for R, F, M
FORECAST_PERIODS = 6              # Months ahead to forecast

# ── Model constants ────────────────────────────────────────────

RANDOM_STATE = 42                 # Reproducibility across all models
TEST_SIZE = 0.2                   # Train/test split ratio

RISK_TIER_THRESHOLDS = {
    "high": 0.7,
    "medium": 0.4,
    "low": 0.0,
}

# ── Data validation boundaries (from Phase 0 inspection) ──────

DATA_START_DATE = "2009-12-01"
DATA_END_DATE = "2011-12-09"
CUSTOMER_ID_MIN = 12346
CUSTOMER_ID_MAX = 18287

# ── ETL performance ───────────────────────────────────────────

BATCH_SIZE = 5000                 # Rows per bulk insert batch


@dataclass
class ETLConfig:
    """Bundled config object passed to ETL classes.

    Exists so we can override values in tests without touching module-level constants.
    Production code uses the defaults; tests can pass modified instances.
    """
    raw_data_path: Path = RAW_DATA_PATH
    sheet_names: list[str] = field(default_factory=lambda: list(SHEET_NAMES))
    database_url: str = DATABASE_URL
    batch_size: int = BATCH_SIZE
    churn_threshold_days: int = CHURN_THRESHOLD_DAYS
    min_valid_price: float = MIN_VALID_PRICE
    quantity_lower_percentile: float = QUANTITY_LOWER_PERCENTILE
    quantity_upper_percentile: float = QUANTITY_UPPER_PERCENTILE
    non_product_stock_codes: set[str] = field(default_factory=lambda: set(NON_PRODUCT_STOCK_CODES))