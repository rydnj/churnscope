"""Churn Prediction: feature engineering, model training, evaluation.

Two classes:
- ChurnFeatureEngineer: builds the feature matrix from the star schema
- ChurnPredictor: trains, evaluates, and compares classification models

Usage:
    python -m backend.analysis.churn_model
"""

from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sqlalchemy import create_engine, text

from backend.etl.config import (
    DATABASE_URL, CHURN_THRESHOLD_DAYS, RANDOM_STATE, TEST_SIZE,
    RISK_TIER_THRESHOLDS,
)


# ── Feature Engineering ───────────────────────────────────────

class ChurnFeatureEngineer:
    """Builds churn feature matrix from the star schema.

    Every feature is computed from fact_transactions + dimensions.
    The churn label is: no purchase in last 90 days relative to
    the max date in the dataset (not today's date).
    """

    def __init__(self, db_engine=None):
        self._engine = db_engine or create_engine(DATABASE_URL)

    def build_features(self) -> pd.DataFrame:
        """Query the star schema and build the full feature matrix.

        Returns:
            DataFrame with one row per customer, feature columns,
            and churn_label (True/False).
        """
        sql = """
        WITH ref AS (
            SELECT MAX(full_date) AS ref_date FROM dim_dates
        ),
        customer_txns AS (
            SELECT
                f.customer_id,
                c.first_purchase,
                c.last_purchase,
                c.region,
                c.acquisition_channel,
                c.age_group,
                d.full_date,
                f.invoice_no,
                f.total_amount,
                f.quantity,
                f.is_return
            FROM fact_transactions f
            JOIN dim_customers c ON f.customer_id = c.customer_id
            JOIN dim_dates d ON f.date_id = d.date_id
        ),
        features AS (
            SELECT
                customer_id,
                -- Recency: days since last purchase
                (SELECT ref_date FROM ref) - MAX(full_date) AS recency_days,

                -- Frequency: distinct non-return orders
                COUNT(DISTINCT CASE WHEN is_return = FALSE THEN invoice_no END) AS frequency,

                -- Monetary: total non-return revenue
                SUM(CASE WHEN is_return = FALSE THEN total_amount ELSE 0 END) AS monetary,

                -- Average order value
                SUM(CASE WHEN is_return = FALSE THEN total_amount ELSE 0 END)
                    / NULLIF(COUNT(DISTINCT CASE WHEN is_return = FALSE THEN invoice_no END), 0)
                    AS avg_order_value,

                -- Return rate: fraction of orders that are returns
                COUNT(DISTINCT CASE WHEN is_return THEN invoice_no END)::FLOAT
                    / NULLIF(COUNT(DISTINCT invoice_no), 0) AS return_rate,

                -- Product diversity: how many distinct products purchased
                COUNT(DISTINCT CASE WHEN is_return = FALSE THEN invoice_no || '-' END) AS order_count,

                -- Days active: span from first to last purchase
                MAX(full_date) - MIN(full_date) AS days_active,

                -- Tenure: days since first purchase to reference date
                (SELECT ref_date FROM ref) - MIN(first_purchase) AS tenure_days,

                -- Items per order (avg quantity per transaction)
                AVG(CASE WHEN is_return = FALSE THEN ABS(quantity) ELSE NULL END) AS avg_items_per_order,

                -- Purchase trend: are they buying more or less over time?
                -- Computed as: orders in second half of tenure minus first half
                -- Positive = accelerating, Negative = decelerating
                COUNT(DISTINCT CASE
                    WHEN is_return = FALSE
                    AND full_date >= first_purchase + (last_purchase - first_purchase) / 2
                    THEN invoice_no END)
                -
                COUNT(DISTINCT CASE
                    WHEN is_return = FALSE
                    AND full_date < first_purchase + (last_purchase - first_purchase) / 2
                    THEN invoice_no END)
                AS purchase_trend,

                -- Demographic features
                MIN(region) AS region,
                MIN(acquisition_channel) AS acquisition_channel,
                MIN(age_group) AS age_group

            FROM customer_txns
            GROUP BY customer_id
        )
        SELECT
            f.*,
            -- Churn label: no purchase in last N days
            CASE
                WHEN f.recency_days > :churn_threshold THEN TRUE
                ELSE FALSE
            END AS churn_label,
            -- Segment from clustering (if available)
            a.segment_name,
            a.cluster_id
        FROM features f
        LEFT JOIN agg_customer_features a ON f.customer_id = a.customer_id
        WHERE f.frequency > 0 AND f.monetary > 0
        """

        with self._engine.connect() as conn:
            df = pd.read_sql(
                text(sql), conn,
                params={"churn_threshold": CHURN_THRESHOLD_DAYS}
            )

        print(f"  Feature matrix: {len(df):,} customers × {len(df.columns)} columns")
        print(f"  Churn rate: {df['churn_label'].mean():.1%}")

        return df

    def get_feature_descriptions(self) -> dict[str, str]:
        """Human-readable description of each feature."""
        return {
            "recency_days": "Days since last purchase (lower = more recent)",
            "frequency": "Number of distinct orders (excluding returns)",
            "monetary": "Total revenue from non-return transactions",
            "avg_order_value": "Average revenue per order",
            "return_rate": "Fraction of orders that were returns",
            "days_active": "Days between first and last purchase",
            "tenure_days": "Days since first purchase to dataset end",
            "avg_items_per_order": "Average number of items per order",
            "region": "Geographic region (categorical)",
            "acquisition_channel": "How customer was acquired (categorical)",
            "age_group": "Customer age bracket (categorical)",
            "segment_name": "RFM cluster segment from Phase 3",
        }


# ── Model Training & Evaluation ──────────────────────────────

class ChurnPredictor:
    """Trains, evaluates, and compares churn classification models."""

    # Numeric features used for modeling
    # NOTE: recency_days is EXCLUDED — it directly encodes the churn label
    # (churn = recency > 90 days). Including it gives perfect but meaningless
    # scores. days_active is also excluded (highly correlated with recency).
    # The model should predict churn from *behavioral patterns*, not from
    # a re-statement of the label.
    NUMERIC_FEATURES = [
        "frequency", "monetary", "avg_order_value",
        "return_rate", "tenure_days", "avg_items_per_order",
        "purchase_trend",
    ]

    # Categorical features to encode
    # NOTE: segment_name excluded — it was built from RFM which includes recency
    CATEGORICAL_FEATURES = ["region", "acquisition_channel", "age_group"]

    def __init__(self):
        self._models: dict = {}
        self._results: dict = {}
        self._scaler = StandardScaler()
        self._label_encoders: dict[str, LabelEncoder] = {}
        self._feature_names: list[str] = []
        self._best_model_name: str | None = None

    def prepare_data(self, df: pd.DataFrame) -> tuple:
        """Split features from label, encode categoricals, scale numerics.

        Returns:
            (X_train, X_test, y_train, y_test)
        """
        # Handle missing values
        df = df.copy()
        for col in self.NUMERIC_FEATURES:
            df[col] = df[col].fillna(0)

        # Encode categoricals
        encoded_cols = []
        for col in self.CATEGORICAL_FEATURES:
            if col in df.columns and df[col].notna().any():
                le = LabelEncoder()
                df[col + "_enc"] = le.fit_transform(df[col].fillna("Unknown").astype(str))
                self._label_encoders[col] = le
                encoded_cols.append(col + "_enc")

        # Build feature matrix
        self._feature_names = self.NUMERIC_FEATURES + encoded_cols
        X = df[self._feature_names].values
        y = df["churn_label"].astype(int).values

        # Scale
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        X_train = self._scaler.fit_transform(X_train)
        X_test = self._scaler.transform(X_test)

        print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
        print(f"  Features: {len(self._feature_names)}")
        print(f"  Train churn rate: {y_train.mean():.1%} | Test churn rate: {y_test.mean():.1%}")

        return X_train, X_test, y_train, y_test

    def train_all(self, X_train, y_train) -> dict:
        """Train all three models.

        Returns:
            Dict of {model_name: fitted_model}
        """
        model_configs = {
            "logistic_regression": LogisticRegression(
                random_state=RANDOM_STATE, max_iter=1000, C=1.0
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_leaf=20,
                random_state=RANDOM_STATE, n_jobs=-1
            ),
        }

        # XGBoost: import only if available
        try:
            from xgboost import XGBClassifier
            model_configs["xgboost"] = XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                min_child_weight=20, random_state=RANDOM_STATE,
                eval_metric="logloss", verbosity=0
            )
        except ImportError:
            print("  ⚠ XGBoost not installed, skipping. Install with: pip install xgboost")

        for name, model in model_configs.items():
            print(f"  Training {name}...")
            model.fit(X_train, y_train)
            self._models[name] = model

        return self._models

    def evaluate_all(self, X_test, y_test) -> pd.DataFrame:
        """Evaluate all trained models and return comparison DataFrame.

        Returns:
            DataFrame with columns: model, precision, recall, f1, auc_roc
        """
        rows = []
        for name, model in self._models.items():
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            auc = roc_auc_score(y_test, y_prob)
            report = classification_report(y_test, y_pred, output_dict=True)

            rows.append({
                "model": name,
                "precision": round(report["1"]["precision"], 4),
                "recall": round(report["1"]["recall"], 4),
                "f1": round(report["1"]["f1-score"], 4),
                "auc_roc": round(auc, 4),
                "accuracy": round(report["accuracy"], 4),
            })

            self._results[name] = {
                "y_pred": y_pred,
                "y_prob": y_prob,
                "auc": auc,
                "report": report,
            }

        results_df = pd.DataFrame(rows).sort_values("auc_roc", ascending=False)
        self._best_model_name = results_df.iloc[0]["model"]

        return results_df

    def get_feature_importance(self, model_name: str | None = None) -> pd.DataFrame:
        """Extract feature importance from a trained model.

        For logistic regression: absolute coefficient values
        For tree models: built-in feature importances

        Returns:
            DataFrame with columns: feature, importance (sorted descending)
        """
        name = model_name or self._best_model_name
        model = self._models[name]

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            return pd.DataFrame()

        fi = pd.DataFrame({
            "feature": self._feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        return fi

    def optimize_threshold(
        self, X_test, y_test,
        model_name: str | None = None,
        cost_fp: float = 10.0,
        cost_fn: float = 100.0,
    ) -> dict:
        """Find the classification threshold that maximizes expected value.

        In business terms:
        - cost_fp: cost of a false positive (sending retention offer to
          someone who wasn't going to churn). E.g., £10 campaign cost.
        - cost_fn: cost of a false negative (missing a churning customer).
          E.g., £100 in lost CLV.

        The optimal threshold minimizes total expected cost.

        Returns:
            Dict with optimal_threshold, expected_value, precision, recall
        """
        name = model_name or self._best_model_name
        y_prob = self._results[name]["y_prob"]

        thresholds = np.arange(0.1, 0.9, 0.01)
        best = {"threshold": 0.5, "cost": float("inf")}

        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            fp = ((y_pred == 1) & (y_test == 0)).sum()
            fn = ((y_pred == 0) & (y_test == 1)).sum()
            total_cost = fp * cost_fp + fn * cost_fn

            if total_cost < best["cost"]:
                best = {
                    "threshold": round(t, 2),
                    "cost": total_cost,
                    "fp": int(fp),
                    "fn": int(fn),
                    "precision": round(float(y_pred[y_pred == 1].sum() and
                        ((y_pred == 1) & (y_test == 1)).sum() / y_pred.sum()), 4),
                    "recall": round(float(
                        ((y_pred == 1) & (y_test == 1)).sum() / y_test.sum()), 4),
                    "f1": round(float(f1_score(y_test, y_pred)), 4),
                }

        return best

    def predict_risk(
        self, X: np.ndarray, customer_ids: pd.Series,
        model_name: str | None = None
    ) -> pd.DataFrame:
        """Predict churn probability and assign risk tiers.

        Returns:
            DataFrame with customer_id, churn_probability, risk_tier
        """
        name = model_name or self._best_model_name
        model = self._models[name]
        probas = model.predict_proba(self._scaler.transform(X) if not isinstance(X, np.ndarray) else X)[:, 1]

        risk_tiers = pd.cut(
            probas,
            bins=[-0.01, RISK_TIER_THRESHOLDS["medium"],
                  RISK_TIER_THRESHOLDS["high"], 1.01],
            labels=["low", "medium", "high"],
        )

        return pd.DataFrame({
            "customer_id": customer_ids.values,
            "churn_probability": probas.round(4),
            "risk_tier": risk_tiers,
        })

    @property
    def best_model_name(self) -> str:
        return self._best_model_name

    # ── Visualization ─────────────────────────────────────────

    def plot_roc_curves(self, y_test, output_dir: Path) -> None:
        """ROC curves for all models on one plot."""
        fig, ax = plt.subplots(figsize=(10, 8))

        for name, res in self._results.items():
            fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
            ax.plot(fpr, tpr, linewidth=2,
                    label=f"{name} (AUC={res['auc']:.4f})")

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUC=0.5)")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves — Churn Prediction Models")
        ax.legend(loc="lower right")

        plt.tight_layout()
        fig.savefig(output_dir / "roc_curves.png")
        plt.close(fig)

    def plot_feature_importance(self, output_dir: Path, top_n: int = 15) -> None:
        """Horizontal bar chart of top feature importances."""
        fi = self.get_feature_importance()
        fi_top = fi.head(top_n)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(
            fi_top["feature"][::-1],
            fi_top["importance"][::-1],
            color="#3498db", alpha=0.8,
        )
        ax.set_xlabel("Importance")
        ax.set_title(f"Top {top_n} Feature Importances ({self._best_model_name})")

        plt.tight_layout()
        fig.savefig(output_dir / "feature_importance.png")
        plt.close(fig)

    def plot_confusion_matrix(self, y_test, output_dir: Path,
                               model_name: str | None = None) -> None:
        """Confusion matrix heatmap."""
        name = model_name or self._best_model_name
        y_pred = self._results[name]["y_pred"]
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Retained", "Churned"],
            yticklabels=["Retained", "Churned"],
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {name}")

        plt.tight_layout()
        fig.savefig(output_dir / f"confusion_matrix_{name}.png")
        plt.close(fig)


# ── CLI Entry Point ───────────────────────────────────────────

def main():
    print("=" * 60)
    print("ChurnScope Churn Prediction")
    print("=" * 60)

    output_dir = Path("reports/churn")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Feature engineering
    print("\n[1/5] Building feature matrix...")
    engineer = ChurnFeatureEngineer()
    features_df = engineer.build_features()

    # Save feature matrix for reference
    features_df.to_csv(output_dir / "feature_matrix.csv", index=False)

    # Step 2: Prepare data
    print("\n[2/5] Preparing train/test split...")
    predictor = ChurnPredictor()
    X_train, X_test, y_train, y_test = predictor.prepare_data(features_df)

    # Step 3: Train models
    print("\n[3/5] Training models...")
    predictor.train_all(X_train, y_train)

    # Step 4: Evaluate
    print("\n[4/5] Evaluating models...")
    comparison = predictor.evaluate_all(X_test, y_test)
    print("\nModel Comparison:")
    print(comparison.to_string(index=False))

    # Feature importance
    print(f"\nBest model: {predictor.best_model_name}")
    fi = predictor.get_feature_importance()
    print("\nTop 10 Feature Importances:")
    print(fi.head(10).to_string(index=False))

    # Threshold optimization
    print("\nThreshold Optimization (cost_fp=£10, cost_fn=£100):")
    threshold_result = predictor.optimize_threshold(X_test, y_test)
    print(f"  Optimal threshold: {threshold_result['threshold']}")
    print(f"  At this threshold — Precision: {threshold_result['precision']}, "
          f"Recall: {threshold_result['recall']}, F1: {threshold_result['f1']}")
    print(f"  False positives: {threshold_result['fp']}, "
          f"False negatives: {threshold_result['fn']}")

    # Step 5: Visualize
    print("\n[5/5] Generating visualizations...")
    predictor.plot_roc_curves(y_test, output_dir)
    predictor.plot_feature_importance(output_dir)
    predictor.plot_confusion_matrix(y_test, output_dir)

    # Save predictions back to database
    print("\nSaving predictions to database...")
    # Re-prepare full dataset for prediction (not just test set)
    df_full = features_df.copy()
    for col in predictor.NUMERIC_FEATURES:
        df_full[col] = df_full[col].fillna(0)
    for col in predictor.CATEGORICAL_FEATURES:
        if col in df_full.columns and col in predictor._label_encoders:
            le = predictor._label_encoders[col]
            df_full[col + "_enc"] = le.transform(df_full[col].fillna("Unknown").astype(str))

    X_full = df_full[predictor._feature_names].values
    X_full_scaled = predictor._scaler.transform(X_full)

    model = predictor._models[predictor.best_model_name]
    probas = model.predict_proba(X_full_scaled)[:, 1]

    engine = create_engine(DATABASE_URL)
    with engine.begin() as conn:
        for cid, prob in zip(df_full["customer_id"], probas):
            risk = "high" if prob >= RISK_TIER_THRESHOLDS["high"] else \
                   "medium" if prob >= RISK_TIER_THRESHOLDS["medium"] else "low"
            conn.execute(text("""
                UPDATE agg_customer_features
                SET churn_probability = :prob,
                    churn_label = :label
                WHERE customer_id = :cid
            """), {"prob": round(float(prob), 4), "label": prob >= 0.5, "cid": int(cid)})

    print(f"  Updated {len(df_full):,} customer predictions in agg_customer_features")

    # Save comparison
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    fi.to_csv(output_dir / "feature_importance.csv", index=False)

    print(f"\n✅ Churn prediction complete. Reports saved to {output_dir}/")


if __name__ == "__main__":
    main()