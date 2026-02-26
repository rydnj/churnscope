-- ============================================================
-- Cohort Retention Analysis
--
-- Groups customers by their first purchase month (cohort),
-- then tracks what percentage are still active N months later.
--
-- This demonstrates: CTEs, window functions, date arithmetic,
-- CROSS JOIN for generating the full cohort grid, and
-- conditional aggregation for retention rates.
-- ============================================================

WITH customer_cohort AS (
    -- Assign each customer to their acquisition cohort (first purchase month)
    SELECT
        c.customer_id,
        DATE_TRUNC('month', c.first_purchase) AS cohort_month
    FROM dim_customers c
),
customer_activity AS (
    -- Get each customer's distinct active months
    SELECT DISTINCT
        f.customer_id,
        DATE_TRUNC('month', d.full_date) AS activity_month
    FROM fact_transactions f
    JOIN dim_dates d ON f.date_id = d.date_id
    WHERE f.is_return = FALSE
),
cohort_data AS (
    -- Join cohort assignment with activity, compute months since acquisition
    SELECT
        cc.cohort_month,
        ca.activity_month,
        -- Months since first purchase (cohort period)
        EXTRACT(YEAR FROM AGE(ca.activity_month, cc.cohort_month)) * 12
        + EXTRACT(MONTH FROM AGE(ca.activity_month, cc.cohort_month)) AS period_number,
        cc.customer_id
    FROM customer_cohort cc
    JOIN customer_activity ca ON cc.customer_id = ca.customer_id
),
cohort_size AS (
    -- Count customers in each cohort
    SELECT
        cohort_month,
        COUNT(DISTINCT customer_id) AS cohort_customers
    FROM customer_cohort
    GROUP BY cohort_month
),
retention AS (
    -- Count retained customers per cohort per period
    SELECT
        cd.cohort_month,
        cd.period_number,
        COUNT(DISTINCT cd.customer_id) AS retained_customers
    FROM cohort_data cd
    WHERE cd.period_number >= 0
    GROUP BY cd.cohort_month, cd.period_number
)
SELECT
    TO_CHAR(r.cohort_month, 'YYYY-MM') AS cohort,
    cs.cohort_customers,
    r.period_number,
    r.retained_customers,
    ROUND(r.retained_customers::NUMERIC / cs.cohort_customers * 100, 2) AS retention_pct
FROM retention r
JOIN cohort_size cs ON r.cohort_month = cs.cohort_month
WHERE r.period_number <= 12    -- Cap at 12 months for readability
ORDER BY r.cohort_month, r.period_number;