-- ============================================================
-- Customer Lifetime Value (CLV) Calculation
--
-- Computes historical CLV per customer and summary statistics
-- by tenure bucket. Used for churn impact estimation:
-- "If we lose a customer from segment X, how much revenue is at risk?"
--
-- Demonstrates: CTEs, date arithmetic, CASE bucketing,
-- aggregate functions, and business-framed metrics.
-- ============================================================

WITH customer_metrics AS (
    SELECT
        f.customer_id,
        c.country,
        c.first_purchase,
        c.last_purchase,
        c.region,
        c.acquisition_channel,
        -- Revenue metrics
        SUM(CASE WHEN f.is_return = FALSE THEN f.total_amount ELSE 0 END) AS gross_revenue,
        SUM(CASE WHEN f.is_return = TRUE THEN f.total_amount ELSE 0 END) AS return_value,
        SUM(f.total_amount) AS net_revenue,
        -- Activity metrics
        COUNT(DISTINCT f.invoice_no) AS total_orders,
        COUNT(DISTINCT CASE WHEN f.is_return THEN f.invoice_no END) AS return_orders,
        COUNT(DISTINCT d.full_date) AS active_days,
        -- Tenure
        (c.last_purchase - c.first_purchase) AS tenure_days,
        -- Average order value
        SUM(CASE WHEN f.is_return = FALSE THEN f.total_amount ELSE 0 END)
            / NULLIF(COUNT(DISTINCT CASE WHEN f.is_return = FALSE THEN f.invoice_no END), 0)
            AS avg_order_value
    FROM fact_transactions f
    JOIN dim_customers c ON f.customer_id = c.customer_id
    JOIN dim_dates d ON f.date_id = d.date_id
    GROUP BY f.customer_id, c.country, c.first_purchase, c.last_purchase,
             c.region, c.acquisition_channel
)
SELECT
    customer_id,
    country,
    region,
    acquisition_channel,
    first_purchase,
    last_purchase,
    tenure_days,
    gross_revenue,
    return_value,
    net_revenue,
    total_orders,
    return_orders,
    ROUND(avg_order_value::NUMERIC, 2) AS avg_order_value,
    -- Monthly revenue rate (for projecting future CLV)
    CASE
        WHEN tenure_days > 30
        THEN ROUND((net_revenue / (tenure_days / 30.0))::NUMERIC, 2)
        ELSE ROUND(net_revenue::NUMERIC, 2)
    END AS monthly_revenue_rate,
    -- Tenure bucket for segmentation
    CASE
        WHEN tenure_days = 0 THEN 'Single Purchase'
        WHEN tenure_days <= 90 THEN '1-3 Months'
        WHEN tenure_days <= 180 THEN '3-6 Months'
        WHEN tenure_days <= 365 THEN '6-12 Months'
        ELSE '12+ Months'
    END AS tenure_bucket
FROM customer_metrics
ORDER BY net_revenue DESC;


-- ============================================================
-- CLV Summary by tenure bucket
-- ============================================================

WITH customer_metrics AS (
    SELECT
        f.customer_id,
        (c.last_purchase - c.first_purchase) AS tenure_days,
        SUM(f.total_amount) AS net_revenue,
        COUNT(DISTINCT f.invoice_no) AS total_orders
    FROM fact_transactions f
    JOIN dim_customers c ON f.customer_id = c.customer_id
    GROUP BY f.customer_id, c.first_purchase, c.last_purchase
),
bucketed AS (
    SELECT
        *,
        CASE
            WHEN tenure_days = 0 THEN 'Single Purchase'
            WHEN tenure_days <= 90 THEN '1-3 Months'
            WHEN tenure_days <= 180 THEN '3-6 Months'
            WHEN tenure_days <= 365 THEN '6-12 Months'
            ELSE '12+ Months'
        END AS tenure_bucket
    FROM customer_metrics
)
SELECT
    tenure_bucket,
    COUNT(*) AS customer_count,
    ROUND(AVG(net_revenue)::NUMERIC, 2) AS avg_clv,
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY net_revenue)::NUMERIC, 2) AS median_clv,
    ROUND(SUM(net_revenue)::NUMERIC, 2) AS total_revenue,
    ROUND(AVG(total_orders)::NUMERIC, 1) AS avg_orders
FROM bucketed
GROUP BY tenure_bucket
ORDER BY avg_clv DESC;