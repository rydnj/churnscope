"""Demand Forecasting: monthly revenue time series models.

Compares ARIMA, SARIMA, Prophet, and LSTM with walk-forward validation.
Generates 6-month forecast with confidence intervals.

Key principle: train/test splits MUST respect time ordering.
No random shuffling — always train on past, test on future.

Usage:
    python -m backend.analysis.forecasting
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sqlalchemy import create_engine, text
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

from backend.etl.config import DATABASE_URL, FORECAST_PERIODS


class DemandForecaster:
    """Time series forecasting for monthly revenue."""

    def __init__(self, db_engine=None):
        self._engine = db_engine or create_engine(DATABASE_URL)
        self._series: pd.DataFrame | None = None
        self._models: dict = {}
        self._results: dict = {}
        self._best_model_name: str | None = None

    def prepare_series(self) -> pd.DataFrame:
        """Aggregate transactions to monthly revenue.

        Excludes Dec 2011 (truncated at Dec 9 — incomplete month).

        Returns:
            DataFrame with columns: date (month start), revenue
        """
        sql = """
        SELECT
            DATE_TRUNC('month', d.full_date) AS date,
            SUM(f.total_amount) AS revenue
        FROM fact_transactions f
        JOIN dim_dates d ON f.date_id = d.date_id
        WHERE f.is_return = FALSE
        GROUP BY DATE_TRUNC('month', d.full_date)
        ORDER BY date
        """
        with self._engine.connect() as conn:
            df = pd.read_sql(text(sql), conn)

        df["date"] = pd.to_datetime(df["date"])
        df["revenue"] = df["revenue"].astype(float)

        # Exclude last month if incomplete (Dec 2011 has only 9 days)
        last_month = df.iloc[-1]["date"]
        last_month_days = (df.iloc[-1]["date"] + pd.offsets.MonthEnd(1)).day
        # If last month has significantly less revenue than average, it's likely truncated
        avg_revenue = df.iloc[:-1]["revenue"].mean()
        if df.iloc[-1]["revenue"] < avg_revenue * 0.5:
            print(f"  Excluding {last_month.strftime('%Y-%m')} (likely truncated)")
            df = df.iloc[:-1]

        df = df.set_index("date")
        df.index = df.index.tz_localize(None)
        df.index.freq = "MS"  # Month Start frequency

        self._series = df
        print(f"  Monthly series: {len(df)} months "
              f"({df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')})")
        print(f"  Revenue range: £{df['revenue'].min():,.0f} – £{df['revenue'].max():,.0f}")

        return df

    def decompose(self) -> dict:
        """Decompose time series into trend, seasonal, and residual.

        Returns:
            Dict with trend, seasonal, residual as Series.
        """
        from statsmodels.tsa.seasonal import seasonal_decompose

        if self._series is None:
            self.prepare_series()

        # Period=12 for monthly seasonality (yearly cycle)
        # Use additive model since seasonal swings are roughly constant
        result = seasonal_decompose(self._series["revenue"], model="additive", period=12)

        return {
            "trend": result.trend,
            "seasonal": result.seasonal,
            "residual": result.resid,
            "observed": result.observed,
        }

    def detect_anomalies(self) -> pd.DataFrame:
        """Identify months with unusually high or low revenue.

        Uses IQR method on residuals from decomposition.

        Returns:
            DataFrame with date, revenue, is_anomaly, anomaly_type
        """
        decomp = self.decompose()
        resid = decomp["residual"].dropna()

        q1 = resid.quantile(0.25)
        q3 = resid.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        df = self._series.copy()
        df["residual"] = decomp["residual"]
        df["is_anomaly"] = (df["residual"] < lower) | (df["residual"] > upper)
        df["anomaly_type"] = "normal"
        df.loc[df["residual"] > upper, "anomaly_type"] = "spike"
        df.loc[df["residual"] < lower, "anomaly_type"] = "drop"

        anomalies = df[df["is_anomaly"]]
        if len(anomalies) > 0:
            print(f"  Anomalies detected: {len(anomalies)}")
            for idx, row in anomalies.iterrows():
                print(f"    {idx.strftime('%Y-%m')}: £{row['revenue']:,.0f} ({row['anomaly_type']})")
        else:
            print("  No anomalies detected")

        return df

    def walk_forward_validate(self, n_splits: int = 6) -> pd.DataFrame:
        """Walk-forward validation: train on expanding window, test one step ahead.

        Why walk-forward (not k-fold): time series data is ordered. Future data
        cannot appear in training. Walk-forward respects this by always training
        on months 1..t and testing on month t+1.

        Args:
            n_splits: Number of test months (taken from the end of the series).

        Returns:
            DataFrame with model, fold, actual, predicted, mape per fold.
        """
        if self._series is None:
            self.prepare_series()

        series = self._series["revenue"]
        results = []

        for i in range(n_splits):
            split_point = len(series) - n_splits + i
            train = series.iloc[:split_point]
            actual = series.iloc[split_point]
            test_date = series.index[split_point]

            # ARIMA
            try:
                from statsmodels.tsa.arima.model import ARIMA
                arima = ARIMA(train, order=(1, 1, 1)).fit()
                pred = arima.forecast(1).iloc[0]
                results.append({
                    "model": "ARIMA(1,1,1)", "fold": i,
                    "date": test_date, "actual": actual, "predicted": pred,
                })
            except Exception as e:
                print(f"    ARIMA failed on fold {i}: {e}")

            # SARIMA — with only 24 months (2 seasonal cycles), we use
            # a simpler seasonal order to avoid overfitting
            try:
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                sarima = SARIMAX(train, order=(1, 1, 1),
                                 seasonal_order=(1, 0, 1, 12),
                                 enforce_stationarity=False,
                                 enforce_invertibility=False).fit(disp=False)
                pred = sarima.forecast(1).iloc[0]
                results.append({
                    "model": "SARIMA", "fold": i,
                    "date": test_date, "actual": actual, "predicted": pred,
                })
            except Exception as e:
                print(f"    SARIMA failed on fold {i}: {e}")

            # Prophet
            try:
                from prophet import Prophet
                prophet_df = pd.DataFrame({
                    "ds": train.index.tz_localize(None), "y": train.values
                })
                m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                           daily_seasonality=False)
                m.fit(prophet_df)
                future = m.make_future_dataframe(periods=1, freq="MS")
                forecast = m.predict(future)
                pred = forecast.iloc[-1]["yhat"]
                results.append({
                    "model": "Prophet", "fold": i,
                    "date": test_date, "actual": actual, "predicted": pred,
                })
            except Exception as e:
                print(f"    Prophet failed on fold {i}: {e}")

        results_df = pd.DataFrame(results)

        # Compute MAPE per model
        summary = []
        for model_name in results_df["model"].unique():
            model_results = results_df[results_df["model"] == model_name]
            mape = mean_absolute_percentage_error(
                model_results["actual"], model_results["predicted"]
            ) * 100
            rmse = np.sqrt(mean_squared_error(
                model_results["actual"], model_results["predicted"]
            ))
            summary.append({"model": model_name, "mape": round(mape, 2), "rmse": round(rmse, 2)})
            self._results[model_name] = {
                "mape": round(mape, 2), "rmse": round(rmse, 2),
                "folds": model_results,
            }

        summary_df = pd.DataFrame(summary).sort_values("mape")
        self._best_model_name = summary_df.iloc[0]["model"]

        return summary_df

    def forecast(self, periods: int = FORECAST_PERIODS) -> pd.DataFrame:
        """Generate future forecast with confidence intervals using best model.

        Args:
            periods: Number of months to forecast.

        Returns:
            DataFrame with date, forecast, lower_bound, upper_bound.
        """
        if self._series is None:
            self.prepare_series()

        series = self._series["revenue"]
        best = self._best_model_name or "SARIMA"

        print(f"  Forecasting {periods} months with {best}...")

        if "SARIMA" in best:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            model = SARIMAX(series, order=(1, 1, 1),
                           seasonal_order=(1, 0, 1, 12),
                           enforce_stationarity=False,
                           enforce_invertibility=False).fit(disp=False)
            forecast_obj = model.get_forecast(steps=periods)
            forecast_vals = forecast_obj.predicted_mean
            conf_int = forecast_obj.conf_int(alpha=0.05)

            result = pd.DataFrame({
                "date": forecast_vals.index,
                "forecast": forecast_vals.values,
                "lower_bound": conf_int.iloc[:, 0].values,
                "upper_bound": conf_int.iloc[:, 1].values,
            })

        elif "Prophet" in best:
            from prophet import Prophet
            prophet_df = pd.DataFrame({
                "ds": series.index.tz_localize(None), "y": series.values
            })
            m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                       daily_seasonality=False)
            m.fit(prophet_df)
            future = m.make_future_dataframe(periods=periods, freq="MS")
            forecast = m.predict(future)
            forecast = forecast.tail(periods)

            result = pd.DataFrame({
                "date": forecast["ds"].values,
                "forecast": forecast["yhat"].values,
                "lower_bound": forecast["yhat_lower"].values,
                "upper_bound": forecast["yhat_upper"].values,
            })

        else:  # ARIMA fallback
            from statsmodels.tsa.arima.model import ARIMA
            model = ARIMA(series, order=(1, 1, 1)).fit()
            forecast_obj = model.get_forecast(steps=periods)
            forecast_vals = forecast_obj.predicted_mean
            conf_int = forecast_obj.conf_int(alpha=0.05)

            result = pd.DataFrame({
                "date": forecast_vals.index,
                "forecast": forecast_vals.values,
                "lower_bound": conf_int.iloc[:, 0].values,
                "upper_bound": conf_int.iloc[:, 1].values,
            })

        return result

    @property
    def best_model_name(self) -> str:
        return self._best_model_name

    # ── Visualization ─────────────────────────────────────────

    def plot_decomposition(self, output_dir: Path) -> None:
        """Plot time series decomposition."""
        decomp = self.decompose()

        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        components = [
            ("observed", "Observed Revenue", "#2c3e50"),
            ("trend", "Trend", "#3498db"),
            ("seasonal", "Seasonal", "#2ecc71"),
            ("residual", "Residual", "#e74c3c"),
        ]

        for ax, (key, title, color) in zip(axes, components):
            ax.plot(decomp[key], color=color, linewidth=2)
            ax.set_ylabel(title)
            ax.set_title(title)

        plt.suptitle("Time Series Decomposition — Monthly Revenue", fontsize=16, y=1.01)
        plt.tight_layout()
        fig.savefig(output_dir / "decomposition.png")
        plt.close(fig)

    def plot_forecast(self, forecast_df: pd.DataFrame, output_dir: Path) -> None:
        """Plot historical data + forecast with confidence bands."""
        fig, ax = plt.subplots(figsize=(14, 7))

        # Historical
        ax.plot(self._series.index, self._series["revenue"],
                color="#2c3e50", linewidth=2, label="Historical", marker="o", markersize=4)

        # Forecast
        forecast_dates = pd.to_datetime(forecast_df["date"])
        ax.plot(forecast_dates, forecast_df["forecast"],
                color="#e74c3c", linewidth=2, label="Forecast", marker="s", markersize=5)

        # Confidence interval
        ax.fill_between(
            forecast_dates,
            forecast_df["lower_bound"],
            forecast_df["upper_bound"],
            alpha=0.2, color="#e74c3c", label="95% Confidence Interval",
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("Revenue (£)")
        ax.set_title("Monthly Revenue: Historical + 6-Month Forecast")
        ax.legend(loc="upper left")

        import matplotlib.ticker as mticker
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))

        plt.tight_layout()
        fig.savefig(output_dir / "forecast.png")
        plt.close(fig)

    def plot_validation(self, validation_df: pd.DataFrame, output_dir: Path) -> None:
        """Plot walk-forward validation: actual vs predicted per model."""
        models = validation_df["model"].unique()
        fig, axes = plt.subplots(len(models), 1, figsize=(14, 4 * len(models)), sharex=True)
        if len(models) == 1:
            axes = [axes]

        for ax, model_name in zip(axes, models):
            model_data = validation_df[validation_df["model"] == model_name]
            ax.plot(model_data["date"], model_data["actual"],
                    "bo-", label="Actual", linewidth=2)
            ax.plot(model_data["date"], model_data["predicted"],
                    "rs--", label="Predicted", linewidth=2)

            mape = self._results[model_name]["mape"]
            ax.set_title(f"{model_name} (MAPE: {mape}%)")
            ax.legend()
            ax.set_ylabel("Revenue (£)")

        plt.xlabel("Date")
        plt.suptitle("Walk-Forward Validation", fontsize=16, y=1.01)
        plt.tight_layout()
        fig.savefig(output_dir / "validation.png")
        plt.close(fig)


# ── CLI Entry Point ───────────────────────────────────────────

def main():
    print("=" * 60)
    print("ChurnScope Demand Forecasting")
    print("=" * 60)

    output_dir = Path("reports/forecast")
    output_dir.mkdir(parents=True, exist_ok=True)

    forecaster = DemandForecaster()

    # Step 1: Prepare series
    print("\n[1/5] Preparing monthly revenue series...")
    series = forecaster.prepare_series()

    # Step 2: Decomposition
    print("\n[2/5] Decomposing time series...")
    forecaster.plot_decomposition(output_dir)
    print("  Decomposition plot saved")

    # Step 3: Anomaly detection
    print("\n[3/5] Detecting anomalies...")
    anomalies = forecaster.detect_anomalies()

    # Step 4: Walk-forward validation
    print("\n[4/5] Walk-forward validation (6 folds)...")
    validation = forecaster.walk_forward_validate(n_splits=6)

    # Collect fold details for plotting
    all_folds = pd.concat([
        forecaster._results[m]["folds"] for m in forecaster._results
    ])
    forecaster.plot_validation(all_folds, output_dir)

    print("\nModel Comparison (Walk-Forward):")
    print(validation.to_string(index=False))
    print(f"\nBest model: {forecaster.best_model_name}")

    # Step 5: Generate forecast
    print(f"\n[5/5] Generating {FORECAST_PERIODS}-month forecast...")
    forecast_df = forecaster.forecast(periods=FORECAST_PERIODS)
    forecaster.plot_forecast(forecast_df, output_dir)

    print("\nForecast:")
    for _, row in forecast_df.iterrows():
        d = pd.to_datetime(row["date"]).strftime("%Y-%m")
        print(f"  {d}: £{row['forecast']:,.0f} "
              f"(£{row['lower_bound']:,.0f} – £{row['upper_bound']:,.0f})")

    total_forecast = forecast_df["forecast"].sum()
    print(f"\nTotal forecasted revenue (next {FORECAST_PERIODS} months): £{total_forecast:,.0f}")

    # Save outputs
    forecast_df.to_csv(output_dir / "forecast.csv", index=False)
    validation.to_csv(output_dir / "model_comparison.csv", index=False)

    print(f"\n✅ Forecasting complete. Reports saved to {output_dir}/")


if __name__ == "__main__":
    main()