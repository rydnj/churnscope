"""RFM Calculator: computes Recency, Frequency, Monetary scores per customer.

Pulls RFM from the star schema via SQL, then exposes the data for
clustering (Phase 3) and churn feature engineering (Phase 4).

Usage:
    python -m backend.analysis.rfm
"""

import pandas as pd
from sqlalchemy import create_engine, text

from backend.etl.config import DATABASE_URL, SQL_DIR


class RFMCalculator:
    """Computes RFM scores per customer from the star schema."""

    def __init__(self, db_engine=None):
        self._engine = db_engine or create_engine(DATABASE_URL)
        self._rfm_df: pd.DataFrame | None = None

    def compute_rfm(self) -> pd.DataFrame:
        """Run the RFM query and return scored DataFrame.

        Returns:
            DataFrame with columns: customer_id, recency_days, frequency,
            monetary, r_score, f_score, m_score, rfm_segment, rfm_total
        """
        sql = (SQL_DIR / "rfm_scores.sql").read_text()

        # The SQL file has two queries separated by semicolons.
        # We only need the main query (everything is one statement here).
        with self._engine.connect() as conn:
            self._rfm_df = pd.read_sql(text(sql), conn)

        print(f"  RFM computed for {len(self._rfm_df):,} customers")
        print(f"  Recency range: {self._rfm_df['recency_days'].min()} – "
              f"{self._rfm_df['recency_days'].max()} days")
        print(f"  Frequency range: {self._rfm_df['frequency'].min()} – "
              f"{self._rfm_df['frequency'].max()} orders")
        print(f"  Monetary range: £{self._rfm_df['monetary'].min():,.2f} – "
              f"£{self._rfm_df['monetary'].max():,.2f}")

        return self._rfm_df

    def get_rfm_summary(self) -> pd.DataFrame:
        """Aggregated stats per RFM total score bucket.

        Useful for a quick view of how customers distribute across
        the scoring spectrum before clustering.
        """
        if self._rfm_df is None:
            self.compute_rfm()

        summary = self._rfm_df.groupby("rfm_total").agg(
            customer_count=("customer_id", "count"),
            avg_recency=("recency_days", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
        ).round(2).reset_index()

        return summary

    @property
    def rfm_data(self) -> pd.DataFrame:
        """Access the computed RFM DataFrame."""
        if self._rfm_df is None:
            self.compute_rfm()
        return self._rfm_df


def main():
    print("=" * 60)
    print("ChurnScope RFM Analysis")
    print("=" * 60)

    calc = RFMCalculator()
    rfm = calc.compute_rfm()

    print("\nRFM Score Distribution:")
    summary = calc.get_rfm_summary()
    print(summary.to_string(index=False))

    print(f"\nTop 10 highest-value customers:")
    top = rfm.nlargest(10, "rfm_total")[
        ["customer_id", "recency_days", "frequency", "monetary", "rfm_segment", "rfm_total"]
    ]
    print(top.to_string(index=False))

    print(f"\nBottom 10 lowest-value customers:")
    bottom = rfm.nsmallest(10, "rfm_total")[
        ["customer_id", "recency_days", "frequency", "monetary", "rfm_segment", "rfm_total"]
    ]
    print(bottom.to_string(index=False))


if __name__ == "__main__":
    main()