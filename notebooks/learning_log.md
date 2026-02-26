# ChurnScope — Learning Log

A running journal of what I built, what I learned, what broke, and why it matters. Written as I go so the reasoning is fresh, not reconstructed after the fact.

---

## Phase 0: Data Inspection

**Date:** Day 1
**Goal:** Understand the raw data before writing any code.

### What I Did

Downloaded the UCI Online Retail II dataset (~1M rows of real UK e-commerce transactions, 2009-2011) and inspected it with a Python script — no cleaning, no modeling, just looking. The script printed column names, dtypes, null counts, value ranges, cardinality, and sample values for both Excel sheets.

### Why This Comes First

Every decision downstream — schema design, cleaning logic, feature engineering — depends on what the data actually looks like. If I'd designed my cleaning pipeline based on assumptions (e.g., "Invoice is probably a clean integer column"), I would've hit a bug that took me hours to trace. Instead, 30 minutes of inspection gave me everything I needed to design correctly the first time.

This is a habit, not a one-time step. In interviews, saying "I inspected the data before designing my schema" signals experience. Juniors jump to modeling. Seniors look at the data first.

### Key Findings

**The dataset has 8 columns across two sheets with identical structure:**

| Column | Dtype | Notable |
|--------|-------|---------|
| Invoice | object | Mixed: numeric strings + "C"-prefixed cancellations |
| StockCode | object | Mixed alphanumeric; includes non-product codes like POST, DOT, BANK CHARGES |
| Description | object | ALL CAPS, has leading/trailing whitespace |
| Quantity | int64 | Range: -80,995 to +80,995 (extreme outliers, std ~200 vs mean ~10) |
| InvoiceDate | datetime64 | Full range: Dec 2009 – Dec 2011 |
| Price | float64 | Range: -53,594 to +38,970 (negatives = adjustments/write-offs) |
| Customer ID | float64 | **~22% null** — the single biggest data quality issue |
| Country | object | ~40 unique, dominated by UK |

**The findings that shaped every downstream decision:**

1. **Customer ID is null in ~22% of rows (~243K rows).** These rows are useless for customer-level analysis (RFM, segmentation, churn) but valid for aggregate revenue and product analysis. So the cleaning strategy needs two paths, not one.

2. **Cancellation invoices start with "C" (~19,494 total), but negative-quantity rows total ~22,950.** They don't perfectly overlap. Some negative quantities aren't flagged as cancellations — likely adjustments or corrections. I used the "C" prefix as the canonical return flag.

3. **The column is called `Customer ID` (with a space), not `CustomerID`.** Tiny detail, would've broken every reference in my code if I assumed otherwise.

4. **Description has leading/trailing whitespace** — e.g., `" WHITE CHERRY LIGHTS"`, `"PINK DOUGHNUT TRINKET POT "`. Needs stripping.

5. **StockCode includes non-product entries** — POST (shipping), DOT (dotcom postage), BANK CHARGES, AMAZONFEE, etc. These aren't products and would pollute any product analysis or RFM calculation.

6. **December 2010 exists in both Excel sheets.** Concatenating without deduplication would double-count an entire month of transactions.

7. **Price goes negative on non-return rows.** These are data entry errors or write-offs, not legitimate transactions.

8. **Quantity outliers are extreme.** A few orders of ±80,995 units with a dataset mean of ~10. These would massively distort any monetary calculation.

### What This Produced

All findings were documented into Section 6 of the groundwork spec — exact column mappings, cleaning rules (in specific order), expected row counts for validation, and constants derived from the data. Nothing gets built until this section is filled. This document became the single source of truth for the entire ETL pipeline.

---

## Phase 1: Database Schema & ETL Pipeline

**Date:** Day 1-2
**Goal:** Build a star schema in PostgreSQL and an idempotent ETL pipeline to populate it.

### Architecture Decisions

**Why PostgreSQL in Docker (not SQLite):**
I have Docker installed, so I ran Postgres as a container from the start. This avoids the classic "works on SQLite but breaks on Postgres" problem when you eventually deploy. The data persists via a Docker volume, so I can stop/start the container without losing anything. When I Dockerize the full app later, this container just moves into `docker-compose.yml` — zero wasted work.

```bash
docker run -d --name churnscope-db \
  -e POSTGRES_USER=churnscope \
  -e POSTGRES_PASSWORD=churnscope_dev \
  -e POSTGRES_DB=churnscope \
  -p 5433:5432 \
  -v churnscope_pgdata:/var/lib/postgresql/data \
  postgres:16
```

Port 5433 because 5432 was already occupied by another project. Same with 8001 for the backend later (8000 was taken).

**Why psycopg v3 (not psycopg2-binary):**
psycopg2-binary still works but psycopg v3 is the actively maintained successor with better async support, which I'll want for FastAPI later. The SQLAlchemy connection string just needs `postgresql+psycopg://` instead of `postgresql://`.

**Why a star schema (not just dumping into one table):**
A star schema separates dimensions (customers, products, dates) from facts (transactions). This optimizes for the analytical query patterns the dashboard needs — aggregations across customers, time, and products — while keeping the ETL idempotent. It also demonstrates data modeling skills that DE/DA interviewers specifically look for.

**Why a staging table:**
Raw cleaned data lands in `stg_transactions` before being split into dimensions and facts. This means I can re-run the dimension/fact loading independently without re-reading the 43MB Excel file. In production pipelines, staging tables are standard practice.

### The Schema

Six tables total:

- **stg_transactions** — cleaned flat data, staging area
- **dim_dates** — one row per calendar date (739 rows for the full date range, including days with zero transactions)
- **dim_products** — one row per unique stock_code (4,905 products)
- **dim_customers** — one row per customer with synthetic demographics (5,894 customers)
- **fact_transactions** — the core fact table with FK references to all dimensions (794,222 rows)
- **agg_customer_features** — ML feature table, populated later in Phases 3-4

Key design choice: `dim_customers` includes `last_purchase` (not in the original spec). Churn is defined as "no purchase in last 90 days" — without this field, every churn query would need to scan the entire fact table.

### The ETL Pipeline

Five modules, each with a single responsibility:

| Module | Responsibility |
|--------|---------------|
| `config.py` | Every constant, threshold, path. Nothing hardcoded elsewhere. |
| `extract.py` | Read Excel sheets, concatenate, validate columns exist. |
| `clean.py` | 10-step cleaning sequence, each step documented with WHY. |
| `transform.py` | Build dimension tables, then fact table with FK mappings. |
| `load.py` | TRUNCATE + INSERT to Postgres, plus validation queries. |
| `pipeline.py` | Orchestrates the flow, runs validation, produces summary. |

The pipeline is idempotent — I can run `python -m backend.etl.pipeline` any number of times and get the same result, because it drops/recreates the schema and truncates tables before loading.

### Cleaning Steps (In Order)

The order matters. Each step depends on the previous one being complete:

1. **Standardize column names** → `Customer ID` becomes `customer_id`
2. **Strip whitespace** from all string columns (cast to str first — this is where a bug lived, see below)
3. **Drop exact duplicates** → removed 34,335 rows (December 2010 sheet overlap)
4. **Flag returns** → invoices starting with "C" get `is_return = True` (flagged, not dropped — returns are valid for return_rate features)
5. **Filter non-product stock codes** → removed 5,631 rows (POST, DOT, BANK CHARGES, etc.)
6. **Filter invalid prices** → removed 5,989 rows (Price ≤ 0 on non-return transactions)
7. **Filter zero quantity** → removed 0 rows (none existed after prior steps)
8. **Cap quantity outliers** at 1st/99th percentile (preserves the transaction, just clips extreme values)
9. **Compute total_amount** = Quantity × Price (negative for returns, which is correct for revenue calculations)
10. **Cast customer_id** to nullable integer (was float64 due to NaN)

Final result: 1,067,371 → 1,021,416 rows (4.31% dropped). Conservative by design.

### Validation

The pipeline runs 8 automatic checks after loading:

- Row counts in Postgres match what was loaded (4 checks, one per table)
- Foreign key integrity: every fact row resolves to valid dimension rows (3 checks)
- No null customer_ids in the fact table (1 check)

All 8 passed on the final run.

### Bugs I Hit and How I Fixed Them

#### Bug 1: Giant SQL statement crashes Postgres

**Symptom:** A wall of text — thousands of parameterized INSERT values printed to the terminal, then a crash.

**Root cause:** `pandas.to_sql()` with `method="multi"` concatenates all rows in a chunk into a single INSERT statement. With a chunk size of 5,000 rows × 9 columns, that's a SQL statement with 45,000+ parameters. Postgres has limits on statement size.

**How I found it:** Ignored the blob of text entirely. Scrolled to the very bottom for the actual exception, then looked at the parameter values near the top. The real error was `NotNullViolation` on `invoice_no`, but the delivery mechanism (giant SQL) was the first thing that had to be fixed.

**Fix:** Removed `method="multi"` and reduced chunk size to 1,000. Pandas default insert method sends one row at a time within a transaction, which is slower but doesn't create monster SQL.

**Lesson:** When you get a wall of text from a database error, read the bottom first (the exception), then search for the column name mentioned in the error. The blob is just the failed SQL being echoed back — it's noise.

#### Bug 2: Every invoice_no was None

**Symptom:** `NotNullViolation: null value in column "invoice_no"` — and every single row had `invoice_no: None`, not just some.

**Root cause:** The raw `Invoice` column has mixed types — some values are actual Python integers (`489434`), some are strings (`"C489434"`). Pandas stores this as `object` dtype. When `_strip_strings()` called `.str.strip()` on the column, it only works on values that are actually strings. The integer values silently became `NaN`. Since most invoices are numeric (non-cancellation), nearly the entire column was wiped out.

**How I found it:** Since *every* row was null (not just some), this was systematic, not a data issue. I traced backwards: the rename in `build_fact_transactions` looked correct, so the problem was earlier. The column was renamed from `Invoice` → `invoice` in `_standardize_columns`, then processed in `_strip_strings` — that's where integers got destroyed.

**Fix:** Changed `_strip_strings()` to cast to `str` first (`df[col].astype(str).str.strip()`), then restore actual NaN for values that were originally null (`"nan"` string back to `None`).

**Lesson:** The debugging pattern for data pipelines is always the same: identify which column/value is wrong from the error message, then trace backwards through the pipeline with print statements or `.head()` calls until you find the step where good data becomes bad data. A quick debug block before the failing step would have caught this instantly:

```python
print(df[["invoice"]].head(10))
print(df["invoice"].isna().sum())
```

### Final Pipeline Output

```
Rows loaded:
  dim_dates: 739
  dim_products: 4,905
  dim_customers: 5,894
  fact_transactions: 794,222
Cleaning: 1,067,371 → 1,021,416 (4.31% dropped)
Duration: 42.19s
All 8 validation checks passed ✅
```

### What I Can Say in an Interview

- "I built an ETL pipeline that ingests 1M+ rows from multi-sheet Excel files, cleans ~45K problematic records across 5 documented steps, transforms into a star schema with 4 tables, and validates referential integrity on every run."
- "I inspected the raw data before designing my schema, which let me catch mixed-type columns and a 22% null rate in customer IDs early, and design my cleaning pipeline around those specific issues."
- "The pipeline is idempotent — safe to re-run any number of times — and completes in under 45 seconds."