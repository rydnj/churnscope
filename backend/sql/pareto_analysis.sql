-- ============================================================
-- Pareto Analysis: Top products by revenue with cumulative %
--
-- Answers: "Do 20% of products generate 80% of revenue?"
-- Classic business analysis question.
--
-- Demonstrates: window functions (SUM OVER, ROW_NUMBER),
-- cumulative distribution, and Pareto principle validation.
-- ============================================================

WITH product_revenue AS (
    SELECT
        p.product_id,
        p.stock_code,
        p.description,
        SUM(f.total_amount) AS revenue,
        SUM(f.quantity) AS units_sold,
        COUNT(DISTINCT f.customer_id) AS unique_buyers
    FROM fact_transactions f
    JOIN dim_products p ON f.product_id = p.product_id
    WHERE f.is_return = FALSE
    GROUP BY p.product_id, p.stock_code, p.description
),
ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (ORDER BY revenue DESC) AS rank,
        SUM(revenue) OVER (ORDER BY revenue DESC) AS cumulative_revenue,
        SUM(revenue) OVER () AS total_revenue
    FROM product_revenue
)
SELECT
    rank,
    stock_code,
    description,
    revenue,
    units_sold,
    unique_buyers,
    cumulative_revenue,
    ROUND(cumulative_revenue / total_revenue * 100, 2) AS cumulative_pct,
    ROUND(rank::NUMERIC / (SELECT COUNT(*) FROM product_revenue) * 100, 2) AS product_pct
FROM ranked
ORDER BY rank;


-- ============================================================
-- Customer-level Pareto: revenue concentration across customers
-- ============================================================

WITH customer_revenue AS (
    SELECT
        f.customer_id,
        SUM(f.total_amount) AS revenue
    FROM fact_transactions f
    WHERE f.is_return = FALSE
    GROUP BY f.customer_id
),
ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (ORDER BY revenue DESC) AS rank,
        SUM(revenue) OVER (ORDER BY revenue DESC) AS cumulative_revenue,
        SUM(revenue) OVER () AS total_revenue,
        COUNT(*) OVER () AS total_customers
    FROM customer_revenue
)
SELECT
    rank,
    customer_id,
    revenue,
    ROUND(cumulative_revenue / total_revenue * 100, 2) AS cumulative_pct,
    ROUND(rank::NUMERIC / total_customers * 100, 2) AS customer_pct
FROM ranked
WHERE ROUND(rank::NUMERIC / total_customers * 100, 2) <= 25
   OR rank <= 20
ORDER BY rank;