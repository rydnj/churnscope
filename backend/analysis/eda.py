"""Exploratory Data Analysis: queries the star schema, generates
statistical findings and visualizations.

Each method answers a specific business question. The generate_report()
method runs everything and saves outputs to a directory.

Usage:
    python -m backend.analysis.eda
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from sqlalchemy import create_engine, text

from backend.etl.config import DATABASE_URL, SQL_DIR


# ── Plot styling ──────────────────────────────────────────────

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "figure.dpi": 150,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


class EDAAnalyzer:
    """Runs all exploratory data analysis against the star schema."""

    def __init__(self, db_engine=None):
        self._engine = db_engine or create_engine(DATABASE_URL)

    def _query(self, sql: str) -> pd.DataFrame:
        """Execute a SQL query and return a DataFrame."""
        with self._engine.connect() as conn:
            return pd.read_sql(text(sql), conn)

    def _read_sql_file(self, filename: str) -> str:
        """Read a .sql file from the sql directory."""
        return (SQL_DIR / filename).read_text()

    # ── Analysis Methods ──────────────────────────────────────

    def revenue_trends(self) -> pd.DataFrame:
        """Monthly revenue with MoM and YoY growth rates.

        Business question: "How is revenue trending? Are we growing?"
        """
        sql = """
        WITH monthly AS (
            SELECT
                d.year, d.month,
                SUM(f.total_amount) AS revenue,
                COUNT(DISTINCT f.customer_id) AS active_customers,
                COUNT(DISTINCT f.invoice_no) AS order_count
            FROM fact_transactions f
            JOIN dim_dates d ON f.date_id = d.date_id
            WHERE f.is_return = FALSE
            GROUP BY d.year, d.month
        )
        SELECT
            year, month, revenue, active_customers, order_count,
            LAG(revenue) OVER (ORDER BY year, month) AS prev_month_revenue,
            ROUND(
                (revenue - LAG(revenue) OVER (ORDER BY year, month))
                / NULLIF(LAG(revenue) OVER (ORDER BY year, month), 0) * 100, 2
            ) AS mom_growth_pct,
            LAG(revenue, 12) OVER (ORDER BY year, month) AS yoy_revenue,
            ROUND(
                (revenue - LAG(revenue, 12) OVER (ORDER BY year, month))
                / NULLIF(LAG(revenue, 12) OVER (ORDER BY year, month), 0) * 100, 2
            ) AS yoy_growth_pct
        FROM monthly
        ORDER BY year, month
        """
        return self._query(sql)

    def cohort_retention(self) -> pd.DataFrame:
        """Customer cohort retention matrix.

        Business question: "How well do we retain customers over time?"
        """
        sql = """
        WITH customer_cohort AS (
            SELECT customer_id, DATE_TRUNC('month', first_purchase) AS cohort_month
            FROM dim_customers
        ),
        customer_activity AS (
            SELECT DISTINCT f.customer_id, DATE_TRUNC('month', d.full_date) AS activity_month
            FROM fact_transactions f
            JOIN dim_dates d ON f.date_id = d.date_id
            WHERE f.is_return = FALSE
        ),
        cohort_data AS (
            SELECT
                cc.cohort_month, ca.activity_month,
                EXTRACT(YEAR FROM AGE(ca.activity_month, cc.cohort_month)) * 12
                + EXTRACT(MONTH FROM AGE(ca.activity_month, cc.cohort_month)) AS period_number,
                cc.customer_id
            FROM customer_cohort cc
            JOIN customer_activity ca ON cc.customer_id = ca.customer_id
        ),
        cohort_size AS (
            SELECT cohort_month, COUNT(DISTINCT customer_id) AS cohort_customers
            FROM customer_cohort GROUP BY cohort_month
        ),
        retention AS (
            SELECT cohort_month, period_number, COUNT(DISTINCT customer_id) AS retained
            FROM cohort_data WHERE period_number >= 0
            GROUP BY cohort_month, period_number
        )
        SELECT
            TO_CHAR(r.cohort_month, 'YYYY-MM') AS cohort,
            cs.cohort_customers,
            r.period_number,
            r.retained,
            ROUND(r.retained::NUMERIC / cs.cohort_customers * 100, 2) AS retention_pct
        FROM retention r
        JOIN cohort_size cs ON r.cohort_month = cs.cohort_month
        WHERE r.period_number <= 12
        ORDER BY r.cohort_month, r.period_number
        """
        return self._query(sql)

    def product_performance(self) -> pd.DataFrame:
        """Top products by revenue with Pareto analysis.

        Business question: "Which products drive our revenue?"
        """
        sql = """
        WITH product_revenue AS (
            SELECT
                p.stock_code, p.description,
                SUM(f.total_amount) AS revenue,
                SUM(f.quantity) AS units_sold,
                COUNT(DISTINCT f.customer_id) AS unique_buyers
            FROM fact_transactions f
            JOIN dim_products p ON f.product_id = p.product_id
            WHERE f.is_return = FALSE
            GROUP BY p.stock_code, p.description
        ),
        ranked AS (
            SELECT *,
                ROW_NUMBER() OVER (ORDER BY revenue DESC) AS rank,
                SUM(revenue) OVER (ORDER BY revenue DESC) AS cumulative_revenue,
                SUM(revenue) OVER () AS total_revenue
            FROM product_revenue
        )
        SELECT
            rank, stock_code, description, revenue, units_sold, unique_buyers,
            ROUND(cumulative_revenue / total_revenue * 100, 2) AS cumulative_pct
        FROM ranked ORDER BY rank
        """
        return self._query(sql)

    def geographic_breakdown(self) -> pd.DataFrame:
        """Revenue and customer distribution by country and region.

        Business question: "Where are our customers and revenue?"
        """
        sql = """
        SELECT
            c.country, c.region,
            COUNT(DISTINCT f.customer_id) AS customers,
            COUNT(DISTINCT f.invoice_no) AS orders,
            SUM(CASE WHEN f.is_return = FALSE THEN f.total_amount ELSE 0 END) AS revenue,
            ROUND(
                SUM(CASE WHEN f.is_return = FALSE THEN f.total_amount ELSE 0 END)
                / NULLIF(COUNT(DISTINCT f.customer_id), 0)::NUMERIC, 2
            ) AS revenue_per_customer
        FROM fact_transactions f
        JOIN dim_customers c ON f.customer_id = c.customer_id
        GROUP BY c.country, c.region
        ORDER BY revenue DESC
        """
        return self._query(sql)

    def churn_statistical_tests(self) -> dict:
        """Statistical tests: are churned customers different from retained?

        Defines churn as no purchase in last 90 days (relative to max date
        in dataset). Tests whether churned vs retained differ on key metrics.

        Business question: "Are churned customers statistically different?"
        """
        sql = """
        WITH customer_metrics AS (
            SELECT
                f.customer_id,
                c.last_purchase,
                (DATE '2011-12-09' - c.last_purchase) AS days_since_last,
                COUNT(DISTINCT f.invoice_no) AS order_count,
                SUM(CASE WHEN f.is_return = FALSE THEN f.total_amount ELSE 0 END) AS revenue,
                AVG(CASE WHEN f.is_return = FALSE THEN f.total_amount ELSE NULL END) AS avg_txn_value,
                COUNT(DISTINCT CASE WHEN f.is_return THEN f.invoice_no END)::FLOAT
                    / NULLIF(COUNT(DISTINCT f.invoice_no), 0) AS return_rate
            FROM fact_transactions f
            JOIN dim_customers c ON f.customer_id = c.customer_id
            GROUP BY f.customer_id, c.last_purchase
        )
        SELECT
            *,
            CASE WHEN days_since_last > 90 THEN TRUE ELSE FALSE END AS is_churned
        FROM customer_metrics
        """
        df = self._query(sql)

        churned = df[df["is_churned"] == True]
        retained = df[df["is_churned"] == False]

        results = {
            "churn_rate": {
                "churned": len(churned),
                "retained": len(retained),
                "total": len(df),
                "churn_pct": round(len(churned) / len(df) * 100, 2),
            },
        }

        # T-tests for continuous variables
        for col in ["order_count", "revenue", "avg_txn_value", "return_rate"]:
            churned_vals = churned[col].dropna()
            retained_vals = retained[col].dropna()

            t_stat, p_value = stats.ttest_ind(retained_vals, churned_vals, equal_var=False)

            results[col] = {
                "retained_mean": round(float(retained_vals.mean()), 2),
                "churned_mean": round(float(churned_vals.mean()), 2),
                "t_statistic": round(float(t_stat), 4),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
                "interpretation": (
                    f"Retained customers have {'higher' if retained_vals.mean() > churned_vals.mean() else 'lower'} "
                    f"{col} (p={'<0.001' if p_value < 0.001 else round(p_value, 4)})"
                ),
            }

        return results

    def customer_tenure_summary(self) -> pd.DataFrame:
        """CLV summary grouped by customer tenure bucket.

        Business question: "How much is each customer segment worth?"
        """
        sql = """
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
            SELECT *,
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
        ORDER BY avg_clv DESC
        """
        return self._query(sql)

    # ── Visualization Methods ─────────────────────────────────

    def plot_revenue_trends(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Monthly revenue bar chart with trend line."""
        fig, ax = plt.subplots(figsize=(14, 6))

        labels = [f"{int(r.year)}-{int(r.month):02d}" for _, r in df.iterrows()]
        colors = ["#e74c3c" if g and g < 0 else "#2ecc71"
                  for g in df["mom_growth_pct"]]

        ax.bar(range(len(df)), df["revenue"], color=colors, alpha=0.8)
        ax.plot(range(len(df)), df["revenue"], color="#2c3e50", linewidth=2, marker="o", markersize=4)

        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_title("Monthly Revenue Trend")
        ax.set_ylabel("Revenue (£)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))

        plt.tight_layout()
        fig.savefig(output_dir / "revenue_trends.png")
        plt.close(fig)

    def plot_cohort_heatmap(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Cohort retention heatmap."""
        pivot = df.pivot_table(
            index="cohort", columns="period_number",
            values="retention_pct", aggfunc="first"
        )
        # Keep only cohorts with enough data
        pivot = pivot[pivot.columns[:13]]

        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(
            pivot, annot=True, fmt=".0f", cmap="YlOrRd_r",
            vmin=0, vmax=100, ax=ax, cbar_kws={"label": "Retention %"}
        )
        ax.set_title("Cohort Retention Analysis (% of Customers Active)")
        ax.set_xlabel("Months Since First Purchase")
        ax.set_ylabel("Cohort (First Purchase Month)")

        plt.tight_layout()
        fig.savefig(output_dir / "cohort_retention.png")
        plt.close(fig)

    def plot_pareto(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Pareto chart: cumulative revenue % by product rank."""
        fig, ax1 = plt.subplots(figsize=(12, 6))

        product_pct = (df["rank"] / len(df) * 100).values
        ax1.fill_between(product_pct, df["cumulative_pct"], alpha=0.3, color="#3498db")
        ax1.plot(product_pct, df["cumulative_pct"], color="#2c3e50", linewidth=2)

        # Mark 80/20 point
        idx_80 = (df["cumulative_pct"] >= 80).idxmax()
        pct_at_80 = product_pct[idx_80]
        ax1.axhline(y=80, color="#e74c3c", linestyle="--", alpha=0.7)
        ax1.axvline(x=pct_at_80, color="#e74c3c", linestyle="--", alpha=0.7)
        ax1.annotate(
            f"{pct_at_80:.1f}% of products = 80% of revenue",
            xy=(pct_at_80, 80), xytext=(pct_at_80 + 10, 70),
            arrowprops=dict(arrowstyle="->", color="#e74c3c"),
            fontsize=11, color="#e74c3c", fontweight="bold",
        )

        ax1.set_xlabel("% of Products (ranked by revenue)")
        ax1.set_ylabel("Cumulative % of Revenue")
        ax1.set_title("Product Revenue Pareto Analysis")
        ax1.set_xlim(0, 100)
        ax1.set_ylim(0, 105)

        plt.tight_layout()
        fig.savefig(output_dir / "pareto_products.png")
        plt.close(fig)

    def plot_geographic(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Revenue by region bar chart."""
        region_df = df.groupby("region").agg(
            revenue=("revenue", "sum"),
            customers=("customers", "sum"),
        ).sort_values("revenue", ascending=True).reset_index()

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(region_df["region"], region_df["revenue"], color="#3498db", alpha=0.8)

        for bar, rev in zip(bars, region_df["revenue"]):
            ax.text(bar.get_width() + 50000, bar.get_y() + bar.get_height() / 2,
                    f"£{rev:,.0f}", va="center", fontsize=10)

        ax.set_xlabel("Revenue (£)")
        ax.set_title("Revenue by Region")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))

        plt.tight_layout()
        fig.savefig(output_dir / "geographic_revenue.png")
        plt.close(fig)

    def plot_stat_tests(self, results: dict, output_dir: Path) -> None:
        """Bar chart comparing churned vs retained customer metrics."""
        metrics = ["order_count", "revenue", "avg_txn_value", "return_rate"]
        labels = ["Order Count", "Revenue (£)", "Avg Transaction", "Return Rate"]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for ax, metric, label in zip(axes, metrics, labels):
            r = results[metric]
            bars = ax.bar(
                ["Retained", "Churned"],
                [r["retained_mean"], r["churned_mean"]],
                color=["#2ecc71", "#e74c3c"], alpha=0.8
            )
            sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else "ns"
            ax.set_title(f"{label} ({sig})")
            ax.set_ylabel(label)

            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=10)

        plt.suptitle("Churned vs Retained Customer Comparison", fontsize=16, y=1.02)
        plt.tight_layout()
        fig.savefig(output_dir / "churn_comparison.png")
        plt.close(fig)

    # ── Report Generator ──────────────────────────────────────

    def generate_report(self, output_dir: str = "reports/eda") -> str:
        """Run all analyses, save visualizations and summary.

        Returns:
            Path to the generated report directory.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print("ChurnScope EDA Report")
        print("=" * 60)

        # 1. Revenue trends
        print("\n[1/6] Revenue trends...")
        revenue_df = self.revenue_trends()
        self.plot_revenue_trends(revenue_df, output_path)
        revenue_df.to_csv(output_path / "revenue_trends.csv", index=False)
        print(f"  Total months: {len(revenue_df)}")
        print(f"  Peak month: {revenue_df.loc[revenue_df['revenue'].idxmax(), 'year']}-"
              f"{int(revenue_df.loc[revenue_df['revenue'].idxmax(), 'month']):02d} "
              f"(£{revenue_df['revenue'].max():,.0f})")
        avg_growth = revenue_df["mom_growth_pct"].dropna().mean()
        print(f"  Avg MoM growth: {avg_growth:.1f}%")

        # 2. Cohort retention
        print("\n[2/6] Cohort retention...")
        cohort_df = self.cohort_retention()
        self.plot_cohort_heatmap(cohort_df, output_path)
        cohort_df.to_csv(output_path / "cohort_retention.csv", index=False)
        # Average retention at month 3
        month3 = cohort_df[cohort_df["period_number"] == 3]["retention_pct"]
        print(f"  Avg 3-month retention: {month3.mean():.1f}%")

        # 3. Product performance
        print("\n[3/6] Product performance...")
        product_df = self.product_performance()
        self.plot_pareto(product_df, output_path)
        product_df.head(50).to_csv(output_path / "top_products.csv", index=False)
        idx_80 = (product_df["cumulative_pct"] >= 80).idxmax()
        pct_products = idx_80 / len(product_df) * 100
        print(f"  Total products: {len(product_df)}")
        print(f"  Pareto: {pct_products:.1f}% of products generate 80% of revenue")

        # 4. Geographic breakdown
        print("\n[4/6] Geographic breakdown...")
        geo_df = self.geographic_breakdown()
        self.plot_geographic(geo_df, output_path)
        geo_df.to_csv(output_path / "geographic_breakdown.csv", index=False)
        uk_pct = geo_df[geo_df["country"] == "United Kingdom"]["revenue"].sum() / geo_df["revenue"].sum() * 100
        print(f"  UK share of revenue: {uk_pct:.1f}%")
        print(f"  Countries with customers: {len(geo_df)}")

        # 5. Statistical tests
        print("\n[5/6] Churn statistical tests...")
        stat_results = self.churn_statistical_tests()
        self.plot_stat_tests(stat_results, output_path)
        print(f"  Overall churn rate: {stat_results['churn_rate']['churn_pct']}%")
        print(f"  Churned: {stat_results['churn_rate']['churned']:,} | "
              f"Retained: {stat_results['churn_rate']['retained']:,}")
        for metric in ["order_count", "revenue", "avg_txn_value", "return_rate"]:
            r = stat_results[metric]
            print(f"  {metric}: {r['interpretation']}")

        # 6. Customer tenure/CLV
        print("\n[6/6] Customer tenure & CLV...")
        tenure_df = self.customer_tenure_summary()
        tenure_df.to_csv(output_path / "tenure_summary.csv", index=False)
        print(tenure_df.to_string(index=False))

        # Save summary
        summary = {
            "total_revenue": float(revenue_df["revenue"].sum()),
            "peak_month": f"{int(revenue_df.loc[revenue_df['revenue'].idxmax(), 'year'])}-"
                          f"{int(revenue_df.loc[revenue_df['revenue'].idxmax(), 'month']):02d}",
            "avg_mom_growth": round(avg_growth, 2),
            "avg_3month_retention": round(float(month3.mean()), 2),
            "pareto_pct_products_for_80_revenue": round(pct_products, 2),
            "uk_revenue_share": round(uk_pct, 2),
            "churn_rate": stat_results["churn_rate"]["churn_pct"],
        }

        import json
        with open(output_path / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n✅ EDA report saved to: {output_path}/")
        print(f"   Files: {len(list(output_path.iterdir()))}")
        return str(output_path)


# ── CLI entry point ───────────────────────────────────────────

def main():
    analyzer = EDAAnalyzer()
    analyzer.generate_report()


if __name__ == "__main__":
    main()