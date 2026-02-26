"""Extractor: reads raw data from source files.

Single responsibility — no cleaning, no transforming. Just read and combine.
This separation means if the data source changes (e.g., from Excel to S3),
only this module needs to change.
"""

import pandas as pd

from backend.etl.config import ETLConfig, RAW_COLUMNS


class Extractor:
    """Reads raw data from source files and returns combined DataFrame."""

    def __init__(self, config: ETLConfig):
        self._config = config

    def extract_transactions(self) -> pd.DataFrame:
        """Read all Excel sheets and concatenate into a single DataFrame.

        Returns:
            Combined DataFrame with original column names, no modifications.
            Columns match RAW_COLUMNS values exactly.

        Raises:
            FileNotFoundError: If raw data file doesn't exist.
        """
        path = self._config.raw_data_path
        if not path.exists():
            raise FileNotFoundError(
                f"Raw data file not found: {path}\n"
                f"Download from: https://archive.ics.uci.edu/dataset/502/online+retail+ii\n"
                f"Place the .xlsx file at: {path}"
            )

        frames: list[pd.DataFrame] = []
        for sheet in self._config.sheet_names:
            print(f"  Reading sheet: {sheet}...")
            df = pd.read_excel(path, sheet_name=sheet)
            print(f"    → {len(df):,} rows")
            frames.append(df)

        combined = pd.concat(frames, ignore_index=True)
        print(f"  Combined total: {len(combined):,} rows")

        self._validate_columns(combined)
        return combined

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Verify all expected raw columns are present.

        Raises:
            ValueError: If expected columns are missing.
        """
        expected = set(RAW_COLUMNS.values())
        actual = set(df.columns)
        missing = expected - actual
        if missing:
            raise ValueError(
                f"Missing expected columns: {missing}\n"
                f"Found columns: {actual}"
            )