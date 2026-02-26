"""Customer Segmentation: K-Means clustering on RFM features.

Takes RFM scores from RFMCalculator, finds optimal K via elbow method
and silhouette analysis, fits clusters, and names the segments based
on their RFM profiles.

Why log-transform before clustering:
    Monetary values in retail are heavily right-skewed — a few whales
    spend 100x the median customer. K-Means uses Euclidean distance,
    so without transformation those whales dominate the clustering
    and you get one "whale" cluster and one "everyone else" cluster.
    Log-transform compresses the scale so all three RFM dimensions
    contribute meaningfully.

Usage:
    python -m backend.analysis.clustering
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sqlalchemy import create_engine, text

from backend.etl.config import DATABASE_URL, RANDOM_STATE
from backend.analysis.rfm import RFMCalculator


class CustomerSegmenter:
    """K-Means clustering on RFM features."""

    FEATURES = ["recency_days", "frequency", "monetary"]

    def __init__(self):
        self._model: KMeans | None = None
        self._scaler = StandardScaler()
        self._k: int | None = None
        self._segment_names: dict[int, str] = {}

    def find_optimal_k(
        self, rfm_df: pd.DataFrame, k_range: range = range(2, 11)
    ) -> dict:
        """Test multiple K values and return inertia + silhouette scores.

        The elbow method looks for where inertia stops dropping sharply.
        Silhouette score measures how well-separated the clusters are
        (higher = better, range -1 to 1).

        You look at both together — the elbow gives a rough range,
        silhouette confirms which K within that range is best.

        Args:
            rfm_df: DataFrame with recency_days, frequency, monetary columns.
            k_range: Range of K values to test.

        Returns:
            Dict of {k: {inertia, silhouette_score}}
        """
        X = self._prepare_features(rfm_df)
        results = {}

        for k in k_range:
            km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
            labels = km.fit_predict(X)
            sil = silhouette_score(X, labels)
            results[k] = {
                "inertia": km.inertia_,
                "silhouette_score": round(sil, 4),
            }
            print(f"  K={k}: inertia={km.inertia_:,.0f}, silhouette={sil:.4f}")

        return results

    def fit(self, rfm_df: pd.DataFrame, k: int) -> "CustomerSegmenter":
        """Fit K-Means with chosen K.

        Args:
            rfm_df: DataFrame with RFM columns.
            k: Number of clusters.

        Returns:
            self (for chaining).
        """
        self._k = k
        X = self._prepare_features(rfm_df)
        self._model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        self._model.fit(X)
        print(f"  Fitted K-Means with K={k}")
        return self

    def predict(self, rfm_df: pd.DataFrame) -> pd.Series:
        """Assign cluster labels to customers.

        Returns:
            Series of cluster IDs (0-indexed).
        """
        if self._model is None:
            raise ValueError("Call fit() first")
        X = self._prepare_features(rfm_df)
        return pd.Series(self._model.predict(X), index=rfm_df.index, name="cluster_id")

    def get_segment_profiles(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """Compute mean RFM values per cluster and assign names.

        Naming logic: based on the relative position of each cluster's
        centroid on R, F, M dimensions. This is where the algorithm's
        output gets translated into business language.

        Returns:
            DataFrame with cluster_id, segment_name, and mean R/F/M values.
        """
        if self._model is None:
            raise ValueError("Call fit() first")

        labels = self.predict(rfm_df)
        rfm_with_labels = rfm_df.copy()
        rfm_with_labels["cluster_id"] = labels

        profiles = rfm_with_labels.groupby("cluster_id").agg(
            customer_count=("customer_id", "count"),
            avg_recency=("recency_days", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
            median_monetary=("monetary", "median"),
            avg_rfm_total=("rfm_total", "mean"),
        ).round(2).reset_index()

        # Auto-name segments based on RFM profile
        self._segment_names = self._auto_name_segments(profiles)
        profiles["segment_name"] = profiles["cluster_id"].map(self._segment_names)

        return profiles

    def get_segment_name(self, cluster_id: int) -> str:
        """Get the human-readable name for a cluster."""
        return self._segment_names.get(cluster_id, f"Segment {cluster_id}")

    @property
    def labels_(self) -> np.ndarray:
        if self._model is None:
            raise ValueError("Call fit() first")
        return self._model.labels_

    @property
    def cluster_centers_(self) -> np.ndarray:
        if self._model is None:
            raise ValueError("Call fit() first")
        return self._model.cluster_centers_

    # ── Private helpers ────────────────────────────────────────

    def _prepare_features(self, rfm_df: pd.DataFrame) -> np.ndarray:
        """Log-transform and standardize RFM features for clustering.

        Why log-transform: monetary is heavily right-skewed. A few customers
        spending £100K+ would dominate Euclidean distance without this.

        Why standardize: R is in days (0-700), F is in counts (1-200),
        M is in pounds (1-300,000). Without scaling, whichever feature
        has the largest range dominates the clustering.
        """
        df = rfm_df[self.FEATURES].copy()

        # Log-transform (add 1 to handle zeros)
        for col in self.FEATURES:
            df[col] = np.log1p(df[col])

        return self._scaler.fit_transform(df)

    def _auto_name_segments(self, profiles: pd.DataFrame) -> dict[int, str]:
        """Assign business names based on RFM centroid positions.

        Sorts clusters by avg_rfm_total (overall value score) and assigns
        names that reflect the business meaning of each profile.
        """
        sorted_profiles = profiles.sort_values("avg_rfm_total", ascending=False)
        names = {}

        for rank, (_, row) in enumerate(sorted_profiles.iterrows()):
            cid = int(row["cluster_id"])
            r = row["avg_recency"]
            f = row["avg_frequency"]
            m = row["avg_monetary"]

            # High value + recent = Champions/Loyalists
            # High value + not recent = At Risk
            # Low value + recent = Promising/New
            # Low value + not recent = Lost/Hibernating

            median_r = profiles["avg_recency"].median()
            median_f = profiles["avg_frequency"].median()
            median_m = profiles["avg_monetary"].median()

            recent = r < median_r
            frequent = f > median_f
            high_spend = m > median_m

            if recent and frequent and high_spend:
                names[cid] = "Champions"
            elif recent and frequent:
                names[cid] = "Loyal Regulars"
            elif recent and high_spend:
                names[cid] = "High-Value Recent"
            elif recent:
                names[cid] = "Promising New"
            elif frequent and high_spend:
                names[cid] = "At-Risk Big Spenders"
            elif frequent:
                names[cid] = "At-Risk Regulars"
            elif high_spend:
                names[cid] = "Fading High-Value"
            else:
                names[cid] = "Lost Low-Value"

        # Deduplicate names if needed (append cluster ID)
        seen = {}
        for cid, name in names.items():
            if name in seen.values():
                names[cid] = f"{name} ({cid})"
            seen[cid] = names[cid]

        return names

    # ── Visualization ─────────────────────────────────────────

    def plot_elbow_silhouette(self, k_results: dict, output_dir: Path) -> None:
        """Plot elbow curve and silhouette scores side by side."""
        ks = list(k_results.keys())
        inertias = [k_results[k]["inertia"] for k in ks]
        silhouettes = [k_results[k]["silhouette_score"] for k in ks]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(ks, inertias, "bo-", linewidth=2)
        ax1.set_xlabel("Number of Clusters (K)")
        ax1.set_ylabel("Inertia")
        ax1.set_title("Elbow Method")

        ax2.plot(ks, silhouettes, "ro-", linewidth=2)
        ax2.set_xlabel("Number of Clusters (K)")
        ax2.set_ylabel("Silhouette Score")
        ax2.set_title("Silhouette Analysis")

        best_k = ks[np.argmax(silhouettes)]
        ax2.axvline(x=best_k, color="green", linestyle="--", alpha=0.7,
                     label=f"Best K={best_k}")
        ax2.legend()

        plt.tight_layout()
        fig.savefig(output_dir / "elbow_silhouette.png")
        plt.close(fig)

    def plot_segment_profiles(
        self, profiles: pd.DataFrame, output_dir: Path
    ) -> None:
        """Radar-style bar chart showing each segment's RFM profile."""
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        metrics = [
            ("avg_recency", "Avg Recency (days)", True),   # lower is better
            ("avg_frequency", "Avg Frequency", False),
            ("avg_monetary", "Avg Monetary (£)", False),
        ]

        for ax, (col, title, invert) in zip(axes, metrics):
            sorted_p = profiles.sort_values(col, ascending=not invert)
            colors = sns.color_palette("viridis", len(sorted_p))
            bars = ax.barh(sorted_p["segment_name"], sorted_p[col], color=colors)
            ax.set_title(title)
            ax.set_xlabel(title)

        plt.suptitle("Customer Segment Profiles", fontsize=16, y=1.02)
        plt.tight_layout()
        fig.savefig(output_dir / "segment_profiles.png")
        plt.close(fig)

    def plot_segment_scatter(
        self, rfm_df: pd.DataFrame, output_dir: Path
    ) -> None:
        """2D scatter plot: Recency vs Monetary, colored by segment."""
        labels = self.predict(rfm_df)
        rfm_plot = rfm_df.copy()
        rfm_plot["segment"] = labels.map(self._segment_names)

        fig, ax = plt.subplots(figsize=(12, 8))
        for segment_name in rfm_plot["segment"].unique():
            mask = rfm_plot["segment"] == segment_name
            ax.scatter(
                rfm_plot.loc[mask, "recency_days"],
                rfm_plot.loc[mask, "monetary"],
                label=segment_name, alpha=0.6, s=30,
            )

        ax.set_xlabel("Recency (days since last purchase)")
        ax.set_ylabel("Monetary (£)")
        ax.set_title("Customer Segments: Recency vs Monetary")
        ax.set_yscale("log")  # Log scale because monetary is skewed
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        fig.savefig(output_dir / "segment_scatter.png")
        plt.close(fig)


# ── CLI entry point ───────────────────────────────────────────

def main():
    print("=" * 60)
    print("ChurnScope Customer Segmentation")
    print("=" * 60)

    output_dir = Path("reports/segmentation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Compute RFM
    print("\n[1/4] Computing RFM scores...")
    rfm_calc = RFMCalculator()
    rfm = rfm_calc.compute_rfm()

    # Step 2: Find optimal K
    print("\n[2/4] Finding optimal K...")
    segmenter = CustomerSegmenter()
    k_results = segmenter.find_optimal_k(rfm)
    segmenter.plot_elbow_silhouette(k_results, output_dir)

    # Pick K: silhouette suggests K=2 but that's not business-useful.
    # When silhouette scores are close (0.35-0.44 range), pick based
    # on business interpretability. K=4 gives actionable segments.
    best_k = 4
    best_sil_k = max(k_results, key=lambda k: k_results[k]["silhouette_score"])
    print(f"\n  Best K by silhouette: {best_sil_k} ({k_results[best_sil_k]['silhouette_score']})")
    print(f"  Chosen K: {best_k} (silhouette {k_results[best_k]['silhouette_score']}) — "
          f"business interpretability over marginal silhouette gain")

    # Step 3: Fit and profile
    print(f"\n[3/4] Fitting K={best_k} and profiling segments...")
    segmenter.fit(rfm, best_k)
    profiles = segmenter.get_segment_profiles(rfm)

    print("\nSegment Profiles:")
    print(profiles[["segment_name", "customer_count", "avg_recency",
                     "avg_frequency", "avg_monetary"]].to_string(index=False))

    # Step 4: Visualize and save
    print("\n[4/4] Generating visualizations...")
    segmenter.plot_segment_profiles(profiles, output_dir)
    segmenter.plot_segment_scatter(rfm, output_dir)

    # Save labels to database
    print("\n[5/5] Saving segment labels...")
    labels = segmenter.predict(rfm)
    rfm_with_segments = rfm.copy()
    rfm_with_segments["cluster_id"] = labels
    rfm_with_segments["segment_name"] = labels.map(
        {k: v for k, v in segmenter._segment_names.items()}
    )

    engine = create_engine(DATABASE_URL)
    with engine.begin() as conn:
        # Update agg_customer_features with RFM + segment data
        for _, row in rfm_with_segments.iterrows():
            conn.execute(text("""
                INSERT INTO agg_customer_features
                    (customer_id, recency_days, frequency, monetary,
                     rfm_segment, cluster_id, segment_name, churn_label)
                VALUES (:cid, :r, :f, :m, :rfm, :cluster, :segment, FALSE)
                ON CONFLICT (customer_id) DO UPDATE SET
                    recency_days = :r, frequency = :f, monetary = :m,
                    rfm_segment = :rfm, cluster_id = :cluster,
                    segment_name = :segment
            """), {
                "cid": int(row["customer_id"]),
                "r": int(row["recency_days"]),
                "f": int(row["frequency"]),
                "m": float(row["monetary"]),
                "rfm": row["rfm_segment"],
                "cluster": int(row["cluster_id"]),
                "segment": row["segment_name"],
            })

    print(f"  Saved {len(rfm_with_segments):,} customer segments to agg_customer_features")

    profiles.to_csv(output_dir / "segment_profiles.csv", index=False)
    rfm_with_segments.to_csv(output_dir / "customer_segments.csv", index=False)

    print(f"\n✅ Segmentation complete. Reports saved to {output_dir}/")


if __name__ == "__main__":
    main()