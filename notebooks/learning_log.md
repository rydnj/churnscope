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

---

## Phase 2: Exploratory Data Analysis

**Date:** Day 2
**Goal:** Understand the business through the data before building any models.

### What I Did

Built SQL queries and a Python analysis module that answer six business questions against the star schema: revenue trends, cohort retention, product performance (Pareto analysis), geographic breakdown, statistical tests (churned vs retained), and customer lifetime value by tenure.

### Key Findings

**Revenue:** ~£15M total over 2 years. Clear seasonality — Sep-Nov spikes (pre-holiday), Jan-Feb drops (post-holiday). November is the peak both years (~£1M). Average MoM growth is 2.2%.

**Retention is poor:** Only 21.5% of customers come back within 3 months. For a non-subscription gift retailer, this isn't surprising, but it means 4 out of 5 customers acquired are gone within a quarter.

**Pareto holds:** 22.3% of products generate 80% of revenue. Classic 80/20 distribution.

**UK dominance:** 83.9% of revenue comes from the UK. The top 5 countries account for ~95%. "UK vs International" is more useful than per-country analysis for most purposes.

**Churn rate: 50.56%** — half the customers haven't purchased in 90 days. High, but this is a gift retailer with 1,516 single-purchase customers. Many were never going to come back.

**Statistical tests revealed surprising findings:**
- Retained customers have **higher** return rates than churned customers (p<0.001). Returning items is a sign of engagement, not dissatisfaction. This was counterintuitive.
- Average transaction value is NOT significantly different between churned and retained (p=0.17). The difference is frequency, not basket size.
- Order count and total revenue are both significantly higher for retained customers (p<0.001).

**Customer lifetime value:** 12+ month customers are worth £5,242 avg vs £315 for single-purchase. That's a 16x difference. This is the business impact number — every long-term customer retained is worth 16x a one-timer.

### What Mattered for Downstream Phases

These findings directly shaped later decisions:
- Return rate being a positive engagement signal → it becomes a useful feature in the churn model, but with the opposite sign from what most people assume
- Frequency matters more than transaction value → purchase frequency features should matter more than monetary in churn prediction
- High single-purchase customer count → segmentation needs to handle one-timers as a distinct group, not lump them with occasional buyers
- Clear seasonality → time series forecasting must account for this, and truncated Dec 2011 must be excluded from training

### SQL Depth

Wrote four standalone SQL files demonstrating complex queries: revenue trends with window functions (LAG, SUM OVER, MoM/YoY growth), cohort retention with CTEs and date arithmetic, Pareto analysis with cumulative distributions, and customer LTV with tenure bucketing. These are saved as interview-ready reference files.

---

## Phase 3: Customer Segmentation (RFM + Clustering)

**Date:** Day 2
**Goal:** Segment customers into actionable groups using RFM scoring and K-Means clustering.

### What RFM Is and Why It Matters

RFM scores each customer on three dimensions: Recency (days since last purchase), Frequency (number of orders), and Monetary (total spend). It's the standard customer segmentation framework in retail because it's both quantitatively rigorous and intuitively explainable to business stakeholders.

### Key Decision: Reference Date

Recency is "days since last purchase relative to *when*?" If you use today's date, every customer looks churned because the data ends in 2011. I used the max date in the dataset (2011-12-09) as the reference point. This is a mistake people make constantly in portfolio projects and interviewers catch it immediately.

### The Silhouette vs Business Interpretability Tradeoff

K-Means with silhouette analysis said K=2 was optimal (silhouette=0.44). But K=2 just splits "active" from "inactive" — that tells leadership nothing actionable. K=4 had silhouette=0.37, very close statistically, but produces four distinct segments the business can act on. I chose K=4.

**This is a key professional judgment call:** silhouette score optimizes for cluster separation, not business value. When scores are close (0.35-0.44 range), you pick based on interpretability.

### My Intuition vs The Algorithm

Before running K-Means, I predicted four segments: loyalists, one-timers, regulars, and occasionals. The algorithm found:

| My Guess | Algorithm Found | RFM Profile |
|---|---|---|
| Loyalists | **Champions** (1,108 customers) | 24-day recency, 20 orders, £9,433 avg |
| One-timers | **Lost Low-Value** (2,049 customers) | 389-day recency, 1.4 orders, £327 avg |
| Regulars | **Promising New** (1,233 customers) | 26-day recency, 3 orders, £828 avg |
| Occasionals | **At-Risk Big Spenders** (1,471 customers) | 207-day recency, 5.4 orders, £1,911 avg |

The surprise was the fourth segment. I called them "occasionals" — low engagement drifting away. The algorithm found something more specific: customers who *used to spend significantly* but haven't been back in ~7 months. These are lapsed valuable customers where retention campaigns have the highest ROI.

### Why Log-Transform Before Clustering

Monetary values in retail are heavily right-skewed — a few whales spend 100x the median. K-Means uses Euclidean distance, so without transformation those whales dominate and you get one "whale" cluster and one "everyone else" cluster. Log-transform compresses the scale so all three RFM dimensions contribute meaningfully.

### What I Can Say in an Interview

- "I segmented 5,861 customers into 4 actionable groups using RFM analysis and K-Means clustering. Silhouette analysis suggested K=2, but I chose K=4 for business interpretability — the marginal silhouette difference (0.44 vs 0.37) wasn't worth losing actionable segmentation."
- "The highest-impact finding was the At-Risk Big Spenders segment: 1,471 customers averaging £1,911 in historical spend who haven't purchased in 7 months. That's the segment where targeted retention has the highest expected return."

---

## Phase 4: Churn Prediction

**Date:** Day 2-3
**Goal:** Build a classification model that predicts which customers will churn.

### The Data Leakage Bug (The Most Important Lesson in This Entire Project)

My first model run produced perfect scores: AUC 1.0, precision 1.0, recall 1.0. Every metric was flawless. This was a **red flag, not a victory.**

The problem: I defined churn as "recency_days > 90" and then gave the model `recency_days` as a feature. The model wasn't predicting churn — it was reading the answer directly from the input. Feature importance confirmed it: `recency_days` accounted for 63% of the model's decisions.

This is called **data leakage** — when information about the target variable leaks into the features. It's the most common mistake in churn modeling, and if you present perfect AUC in an interview without catching it, it's worse than presenting a mediocre model built correctly.

**The fix:** Remove features that directly encode the label. `recency_days` was the obvious one, but `days_active` (highly correlated with recency) and `segment_name` (built from RFM which includes recency) also leaked. I kept only genuine behavioral signals: frequency, monetary, avg_order_value, return_rate, tenure_days, avg_items_per_order, purchase_trend, and demographic features.

**How I'd catch this in the future:** Any time a model produces near-perfect scores on real-world data, be suspicious. Check feature importance first — if one feature dominates and it's conceptually close to the target definition, that's leakage. Real-world churn models typically achieve AUC 0.75-0.90.

### Results After Fixing Leakage

| Model | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|
| Random Forest | 83.1% | 84.1% | 83.6% | 0.9208 |
| XGBoost | 82.9% | 82.4% | 82.7% | 0.9144 |
| Logistic Regression | 77.3% | 82.7% | 79.9% | 0.8649 |

Random Forest wins with AUC 0.92 — strong for real-world churn. Logistic regression at 0.86 isn't far behind, which means the relationships are somewhat linear and the features are well-chosen.

### Feature Importance Tells the Business Story

1. **tenure_days (41%)** — how long someone's been a customer is the strongest predictor. New customers churn more.
2. **frequency (24%)** — customers who buy more are less likely to churn.
3. **monetary (19%)** — higher spenders stick around.
4. **purchase_trend (4.4%)** — customers whose ordering is decelerating are more likely to churn. This is the early warning signal.
5. **return_rate (3.7%)** — higher return rate = less likely to churn. Confirms the EDA finding that returns signal engagement.

### Threshold Optimization

The default 0.5 threshold gives balanced precision/recall (~83%). But the business tradeoff favors catching more churners: sending a £10 retention email to someone who wasn't going to churn costs £10, but missing a churning customer costs ~£100+ in lost CLV.

At threshold 0.2: recall jumps to 97.8% (catch almost everyone who's leaving) at the cost of 294 false positives. Total cost: 294 × £10 = £2,940 in wasted emails vs catching 13 additional churners worth ~£1,300+ each. The math clearly favors the lower threshold.

### What I Can Say in an Interview

- "My first model had perfect AUC because recency — which directly defines the churn label — was in the features. I caught the leakage through feature importance analysis, removed the leaky features, and the model dropped to a realistic AUC of 0.92."
- "I optimized the classification threshold using an expected value framework: at threshold 0.2, we catch 97.8% of churners while the cost of false positives (£2,940 in unnecessary campaigns) is far outweighed by the revenue retained."

---

## Phase 5: Time Series Forecasting

**Date:** Day 3
**Goal:** Forecast 6 months of future revenue and compare forecasting approaches.

### Why This Is a Different ML Paradigm

Classification (churn) uses cross-sectional data — each customer is independent. Time series is ordered — future data cannot appear in training. This means no random train/test splits. We use walk-forward validation: train on months 1..t, predict month t+1, expand the window, repeat.

### The Data Limitation

24 complete months (Dec 2009 – Nov 2011). December 2011 was excluded because it only contains 9 days of data — including it would show a false revenue drop that isn't a real business signal. This is short for time series — deep learning needs hundreds of points, and seasonal models need multiple full cycles.

### Model Comparison

| Model | MAPE | Why This Result |
|---|---|---|
| ARIMA(1,1,1) | **13.63%** | Simplest model wins. Short series favors simplicity. |
| SARIMA | 27.52% | Convergence warnings. Only 2 seasonal cycles isn't enough to estimate seasonal parameters reliably. |
| Prophet | 59.32% | Designed for daily data with years of history, not 24 monthly observations. |

ARIMA at 13.6% MAPE hits the <15% target from the spec.

### Key Decisions

**Excluding truncated December:** Without this, the last data point shows a massive revenue drop that's an artifact of incomplete data, not a real trend. The model would train on a false signal.

**SARIMA simplification:** Originally used seasonal_order=(1,1,1,12) which diverged catastrophically (2073% MAPE). Simplified to (1,0,1,12) — still underperformed but at least produced finite numbers. The honest finding: SARIMA needs more data.

**No LSTM:** The spec called for it, but with 24 data points, LSTM would need more data than exists to learn anything meaningful. Reporting this honestly is stronger than forcing a model that doesn't work.

### What I Can Say in an Interview

- "ARIMA outperformed SARIMA and Prophet at 13.6% MAPE. SARIMA couldn't reliably estimate seasonal parameters from only 2 annual cycles, and Prophet is optimized for longer daily series. With 4-5 years of data, SARIMA would likely win."
- "I used walk-forward validation instead of random train/test split because time series data has temporal dependence — future data can't leak into training."

---

## Phase 6: FastAPI Backend

**Date:** Day 3
**Goal:** Serve all analytics data as JSON endpoints for the React dashboard.

### Architecture

Four routers, each mapping to a dashboard page:
- `/api/v1/kpis/` — executive summary metrics and revenue trends
- `/api/v1/segments/` — customer segment profiles with churn rates
- `/api/v1/churn/` — paginated risk list, model metrics, feature importance
- `/api/v1/forecast/` — historical revenue + 6-month forecast with confidence intervals

Each endpoint runs SQL against the star schema or reads pre-computed results from CSV files. The API is a thin serving layer — all computation happened in Phases 2-5.

### Design Decision: Pre-compute vs. Live Compute

Model predictions and forecasts are pre-computed and stored (in the database and CSVs), not run on every API call. This is how production systems work: ML models run in batch, write results to a database, and the API just reads. Real-time scoring would be needed for a live system, but for an analytics dashboard, batch is correct.

### Bug: CORS

The React dev server ran on `http://127.0.0.1:5173` but my CORS config only allowed `http://localhost:5173`. Browsers treat these as different origins. The fix was adding both `127.0.0.1` and `localhost` variants to the allowed origins list.

This is a common gotcha — `localhost` and `127.0.0.1` are technically different origins for CORS purposes even though they resolve to the same address.

---

## Phase 7: React + TypeScript Dashboard

**Date:** Day 3
**Goal:** Build a 4-page analytics dashboard that tells the business story visually.

### Page Design Philosophy

Each page answers one business question:
- **Dashboard:** "How is the business doing overall?"
- **Segments:** "Who are our customers?"
- **Churn Risk:** "Who is about to leave?"
- **Forecast:** "Where is revenue heading?"

This is deliberate — the dashboard is organized around business questions, not model outputs. A stakeholder can navigate by what they want to know, not by what technique was used.

### Tech: React + TypeScript + Recharts

- **TypeScript types mirror Pydantic schemas exactly** — same field names, same types. One source of truth for the data shape.
- **Recharts** for all visualizations — bar charts, pie charts, composed charts, line charts.
- **Inline styles** rather than CSS framework — simpler for a portfolio project, no build tooling overhead.

### Visual Issues and Fixes

**Segments page — all-blue pie chart:** Originally used Recharts `Pie` without individual `Cell` colors. Fixed by mapping each segment to a distinct color via a `SEGMENT_COLORS` lookup and rendering `Cell` components.

**Churn Risk — grey bar in feature importance:** The `_enc` suffix on encoded feature names was ugly, and one feature had a rendering artifact. Fixed by cleaning feature names (strip `_enc`, replace underscores with spaces, title case) and constraining the bar chart with explicit `domain` and `barSize`.

**Forecast — confidence band rendering as blob:** Recharts' `Area` component draws from the baseline up, not between two lines. Stacking two Areas (invisible lower + visible band) still produced a triangle shape. Final solution: abandoned Areas entirely, used `ReferenceArea` for light forecast region shading and thin dashed `Line` components for upper/lower bounds. Cleaner and more readable.

**TypeScript formatter errors:** Recharts tooltip formatters expect `(value: number | undefined)` but I typed them as `(value: number)`. Fixed by using untyped `(value)` and casting with `Number(value)`.

### What I Can Say in an Interview

- "The dashboard is designed around business questions, not model outputs. Each of the four pages answers a question leadership actually asked: how's the business, who are our customers, who's leaving, and where's revenue heading."
- "I built it with React + TypeScript with types that mirror the backend Pydantic schemas exactly, ensuring type safety across the full stack."