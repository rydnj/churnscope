-- ============================================================
-- Revenue Trends: Monthly revenue with running total, MoM growth,
-- and YoY comparison using window functions.
--
-- This query demonstrates: window functions (LAG, SUM OVER),
-- CTEs, date aggregation, and growth rate calculation.
-- ============================================================

WITH monthly AS (
    SELECT
        d.year,
        d.month,
        SUM(f.total_amount) AS revenue,
        COUNT(DISTINCT f.customer_id) AS active_customers,
        COUNT(DISTINCT f.invoice_no) AS order_count,
        SUM(CASE WHEN f.is_return THEN 1 ELSE 0 END) AS return_count
    FROM fact_transactions f
    JOIN dim_dates d ON f.date_id = d.date_id
    WHERE f.is_return = FALSE
    GROUP BY d.year, d.month
),
with_growth AS (
    SELECT
        year,
        month,
        revenue,
        active_customers,
        order_count,
        return_count,
        -- Running total
        SUM(revenue) OVER (ORDER BY year, month) AS cumulative_revenue,
        -- Month-over-month growth
        LAG(revenue) OVER (ORDER BY year, month) AS prev_month_revenue,
        ROUND(
            (revenue - LAG(revenue) OVER (ORDER BY year, month))
            / NULLIF(LAG(revenue) OVER (ORDER BY year, month), 0) * 100,
            2
        ) AS mom_growth_pct,
        -- Year-over-year comparison (12-month lag)
        LAG(revenue, 12) OVER (ORDER BY year, month) AS yoy_revenue,
        ROUND(
            (revenue - LAG(revenue, 12) OVER (ORDER BY year, month))
            / NULLIF(LAG(revenue, 12) OVER (ORDER BY year, month), 0) * 100,
            2
        ) AS yoy_growth_pct
    FROM monthly
)
SELECT * FROM with_growth
ORDER BY year, month;


-- ============================================================
-- Quarterly revenue summary
-- ============================================================

SELECT
    d.year,
    d.quarter,
    SUM(f.total_amount) AS revenue,
    COUNT(DISTINCT f.customer_id) AS active_customers,
    COUNT(DISTINCT f.invoice_no) AS order_count,
    ROUND(SUM(f.total_amount) / COUNT(DISTINCT f.customer_id), 2) AS revenue_per_customer
FROM fact_transactions f
JOIN dim_dates d ON f.date_id = d.date_id
WHERE f.is_return = FALSE
GROUP BY d.year, d.quarter
ORDER BY d.year, d.quarter;