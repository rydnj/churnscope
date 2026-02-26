-- ChurnScope Star Schema
-- Run with: psql -U churnscope -d churnscope -f backend/sql/schema.sql
-- Idempotent: DROP IF EXISTS ensures re-runnability

-- ============================================================
-- Drop in reverse dependency order (facts before dimensions)
-- ============================================================

DROP TABLE IF EXISTS agg_customer_features CASCADE;
DROP TABLE IF EXISTS fact_transactions CASCADE;
DROP TABLE IF EXISTS dim_customers CASCADE;
DROP TABLE IF EXISTS dim_products CASCADE;
DROP TABLE IF EXISTS dim_dates CASCADE;

-- ============================================================
-- Staging table: raw combined data before star schema transform
-- ============================================================

DROP TABLE IF EXISTS stg_transactions;

CREATE TABLE stg_transactions (
    invoice         VARCHAR(20) NOT NULL,
    stock_code      VARCHAR(20) NOT NULL,
    description     VARCHAR(500),
    quantity         INTEGER NOT NULL,
    invoice_date    TIMESTAMP NOT NULL,
    price           NUMERIC(10,2) NOT NULL,
    customer_id     INTEGER,               -- nullable: ~22% of raw data
    country         VARCHAR(100) NOT NULL,
    is_return       BOOLEAN NOT NULL DEFAULT FALSE,
    total_amount    NUMERIC(12,2) NOT NULL
);

-- ============================================================
-- Dimension: Dates
-- ============================================================

CREATE TABLE dim_dates (
    date_id         SERIAL PRIMARY KEY,
    full_date       DATE NOT NULL UNIQUE,
    day_of_week     SMALLINT NOT NULL,      -- 0=Monday, 6=Sunday (Python convention)
    day_name        VARCHAR(10) NOT NULL,    -- Monday, Tuesday, etc.
    month           SMALLINT NOT NULL,
    month_name      VARCHAR(10) NOT NULL,
    quarter         SMALLINT NOT NULL,
    year            SMALLINT NOT NULL,
    is_holiday      BOOLEAN NOT NULL DEFAULT FALSE,
    is_weekend      BOOLEAN NOT NULL DEFAULT FALSE
);

-- ============================================================
-- Dimension: Customers
-- ============================================================

CREATE TABLE dim_customers (
    customer_id         INTEGER PRIMARY KEY,
    country             VARCHAR(100) NOT NULL,
    first_purchase      DATE NOT NULL,
    last_purchase       DATE NOT NULL,
    -- Synthetic demographic fields (generated during ETL)
    age_group           VARCHAR(20),
    region              VARCHAR(50),
    acquisition_channel VARCHAR(50)
);

-- ============================================================
-- Dimension: Products
-- ============================================================

CREATE TABLE dim_products (
    product_id      SERIAL PRIMARY KEY,
    stock_code      VARCHAR(20) NOT NULL UNIQUE,
    description     VARCHAR(500),
    category        VARCHAR(100),           -- derived later via description clustering
    avg_unit_price  NUMERIC(10,2)
);

-- ============================================================
-- Fact: Transactions
-- ============================================================

CREATE TABLE fact_transactions (
    transaction_id  SERIAL PRIMARY KEY,
    invoice_no      VARCHAR(20) NOT NULL,
    customer_id     INTEGER NOT NULL REFERENCES dim_customers(customer_id),
    product_id      INTEGER NOT NULL REFERENCES dim_products(product_id),
    date_id         INTEGER NOT NULL REFERENCES dim_dates(date_id),
    quantity         INTEGER NOT NULL,
    unit_price      NUMERIC(10,2) NOT NULL,
    total_amount    NUMERIC(12,2) NOT NULL,
    is_return       BOOLEAN NOT NULL DEFAULT FALSE
);

-- ============================================================
-- Aggregate: Customer Features (for ML and API serving)
-- Populated after analysis phases (RFM, clustering, churn model)
-- ============================================================

CREATE TABLE agg_customer_features (
    customer_id         INTEGER PRIMARY KEY REFERENCES dim_customers(customer_id),
    recency_days        INTEGER NOT NULL,
    frequency           INTEGER NOT NULL,
    monetary            NUMERIC(12,2) NOT NULL,
    avg_order_value     NUMERIC(10,2),
    return_rate         NUMERIC(5,4),       -- 0.0000 to 1.0000
    product_diversity   INTEGER,
    days_active         INTEGER,
    purchase_trend      NUMERIC(8,4),       -- slope of monthly purchase frequency
    rfm_segment         VARCHAR(20),        -- R/F/M quintile label (e.g., "5-4-5")
    cluster_id          INTEGER,
    segment_name        VARCHAR(50),        -- human-readable segment name
    churn_label         BOOLEAN,            -- no purchase in last 90 days
    churn_probability   NUMERIC(5,4)        -- model prediction 0.0000 to 1.0000
);

-- ============================================================
-- Indexes for query performance
-- ============================================================

-- Fact table: these cover the primary query patterns
CREATE INDEX idx_fact_txn_customer_id ON fact_transactions(customer_id);
CREATE INDEX idx_fact_txn_product_id ON fact_transactions(product_id);
CREATE INDEX idx_fact_txn_date_id ON fact_transactions(date_id);
CREATE INDEX idx_fact_txn_invoice_no ON fact_transactions(invoice_no);
CREATE INDEX idx_fact_txn_is_return ON fact_transactions(is_return);

-- Dimension tables
CREATE INDEX idx_dim_dates_full_date ON dim_dates(full_date);
CREATE INDEX idx_dim_dates_year_month ON dim_dates(year, month);
CREATE INDEX idx_dim_customers_country ON dim_customers(country);
CREATE INDEX idx_dim_products_stock_code ON dim_products(stock_code);

-- Aggregate table: dashboard query patterns
CREATE INDEX idx_agg_cf_segment ON agg_customer_features(segment_name);
CREATE INDEX idx_agg_cf_churn_label ON agg_customer_features(churn_label);
CREATE INDEX idx_agg_cf_churn_prob ON agg_customer_features(churn_probability DESC);

-- Staging table: for dedup during ETL
CREATE INDEX idx_stg_txn_invoice ON stg_transactions(invoice);
CREATE INDEX idx_stg_txn_customer_id ON stg_transactions(customer_id);