-- ============================================================
-- RFM Score Calculation
--
-- Computes Recency, Frequency, Monetary per customer,
-- then assigns quintile scores (1-5) using NTILE.
--
-- Reference date is the max invoice date in the dataset,
-- NOT today's date — because the data ends in Dec 2011.
-- Using today's date would make every customer look churned.
--
-- NTILE(5) splits customers into 5 equal-sized groups per metric.
-- For Recency, 5 = most recent (best), 1 = least recent (worst).
-- For Frequency and Monetary, 5 = highest (best), 1 = lowest (worst).
-- ============================================================

WITH reference AS (
    SELECT MAX(full_date) AS ref_date FROM dim_dates
),
customer_rfm AS (
    SELECT
        f.customer_id,
        -- Recency: days since last purchase
        (SELECT ref_date FROM reference) - MAX(d.full_date) AS recency_days,
        -- Frequency: number of distinct orders (excluding returns)
        COUNT(DISTINCT CASE WHEN f.is_return = FALSE THEN f.invoice_no END) AS frequency,
        -- Monetary: total non-return revenue
        SUM(CASE WHEN f.is_return = FALSE THEN f.total_amount ELSE 0 END) AS monetary
    FROM fact_transactions f
    JOIN dim_dates d ON f.date_id = d.date_id
    GROUP BY f.customer_id
),
rfm_scored AS (
    SELECT
        customer_id,
        recency_days,
        frequency,
        ROUND(monetary::NUMERIC, 2) AS monetary,
        -- Recency: REVERSE scored — lower days = better = higher score
        NTILE(5) OVER (ORDER BY recency_days DESC) AS r_score,
        -- Frequency: higher = better = higher score
        NTILE(5) OVER (ORDER BY frequency ASC) AS f_score,
        -- Monetary: higher = better = higher score
        NTILE(5) OVER (ORDER BY monetary ASC) AS m_score
    FROM customer_rfm
    WHERE frequency > 0 AND monetary > 0
)
SELECT
    *,
    r_score || '-' || f_score || '-' || m_score AS rfm_segment,
    r_score + f_score + m_score AS rfm_total
FROM rfm_scored
ORDER BY rfm_total DESC;