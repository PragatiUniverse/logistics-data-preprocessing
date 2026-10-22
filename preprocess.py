
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INPUT_FILE = DATA_DIR / "logistics_orders_raw.csv"
OUTPUT_FILE = DATA_DIR / "logistics_orders_cleaned.csv"
SUMMARY_FILE = DATA_DIR / "data_quality_summary.csv"

def main():
    print("=" * 55)
    print("LOGISTICS DATA CLEANING AND PREPROCESSING")
    print("=" * 55)

    # 1. Check input file
    if not INPUT_FILE.exists():
        print(f"\nERROR: Input file not found:\n{INPUT_FILE}")
        print("Place logistics_orders_raw.csv inside the data folder.")
        return

    # 2. Load data
    df = pd.read_csv(INPUT_FILE)
    print(f"\nOriginal dataset shape: {df.shape}")
    print("\nFirst five rows:")
    print(df.head().to_string(index=False))

    # 3. Initial data quality checks
    original_rows = len(df)
    missing_before = int(df.isna().sum().sum())
    duplicates_before = int(df.duplicated().sum())

    print(f"\nMissing cells before cleaning: {missing_before}")
    print(f"Duplicate rows before cleaning: {duplicates_before}")

    # 4. Normalize text and empty cells
    df = df.replace(r"^\s*$", np.nan, regex=True)

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].map(
            lambda x: x.strip() if isinstance(x, str) else x
        )

    # Standardize common text fields
    for col in ["status", "delivery_status", "mode", "transport_mode"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.title()

    # 5. Convert date columns
    date_columns = [
        col for col in df.columns
        if "date" in col.lower() or col.lower() in ["order_time", "delivery_time"]
    ]
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # 6. Convert likely numeric columns
    numeric_keywords = [
        "distance", "weight", "cost", "price", "rating",
        "quantity", "days", "duration"
    ]
    for col in df.columns:
        if any(word in col.lower() for word in numeric_keywords):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 7. Remove exact duplicate records
    df = df.drop_duplicates().copy()

    # 8. Flag invalid numeric values
    for col in df.columns:
        name = col.lower()
        if any(word in name for word in ["distance", "weight", "cost", "price"]):
            df.loc[df[col] < 0, col] = np.nan

    for col in df.columns:
        if "rating" in col.lower():
            df.loc[~df[col].between(1, 5), col] = np.nan

    # 9. Report outliers using the IQR method (do not delete them)
    outlier_counts = {}
    for col in df.select_dtypes(include="number").columns:
        series = df[col].dropna()
        if len(series) < 4:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            outlier_mask = (df[col] < q1 - 1.5 * iqr) | (
                df[col] > q3 + 1.5 * iqr
            )
            outlier_counts[col] = int(outlier_mask.sum())

    # 10. Preserve missing-rating information, then impute ratings
    for col in df.columns:
        if "rating" in col.lower():
            df[f"{col}_missing"] = df[col].isna().astype(int)
            if df[col].notna().any():
                df[col] = df[col].fillna(df[col].median())

    # 11. Remove rows missing essential order identifiers
    essential_columns = [
        col for col in df.columns
        if col.lower() in ["order_id", "order_date", "origin", "destination"]
    ]
    if essential_columns:
        df = df.dropna(subset=essential_columns).copy()

    # 12. Feature engineering for delivery dates, when columns exist
    lower_to_actual = {col.lower(): col for col in df.columns}

    order_date_col = lower_to_actual.get("order_date")
    delivery_date_col = lower_to_actual.get("delivery_date")
    promised_date_col = lower_to_actual.get("promised_date")

    if order_date_col and delivery_date_col:
        df["delivery_days"] = (
            df[delivery_date_col] - df[order_date_col]
        ).dt.days
        df.loc[df["delivery_days"] < 0, "delivery_days"] = np.nan

    if order_date_col and promised_date_col:
        df["promised_days"] = (
            df[promised_date_col] - df[order_date_col]
        ).dt.days
        df.loc[df["promised_days"] < 0, "promised_days"] = np.nan

    if delivery_date_col and promised_date_col:
        df["delay_days"] = (
            df[delivery_date_col] - df[promised_date_col]
        ).dt.days
        df["late_delivery"] = np.where(
            df[delivery_date_col].notna(),
            (df["delay_days"] > 0).astype(int),
            np.nan
        )

    # 13. Scale selected numeric columns into 0-to-1 copies
    scale_columns = [
        col for col in df.columns
        if any(word in col.lower() for word in ["distance", "weight", "cost"])
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    for col in scale_columns:
        if df[col].notna().any():
            values = df[col].fillna(df[col].median()).to_numpy().reshape(-1, 1)
            scaler = MinMaxScaler()
            df[f"{col}_scaled"] = scaler.fit_transform(values).ravel()

    # 14. Save cleaned dataset
    df.to_csv(OUTPUT_FILE, index=False)

    # 15. Save data quality summary
    summary = pd.DataFrame({
        "metric": [
            "original_rows",
            "cleaned_rows",
            "duplicate_rows_removed",
            "missing_cells_before_cleaning",
            "missing_cells_after_cleaning",
        ],
        "value": [
            original_rows,
            len(df),
            duplicates_before,
            missing_before,
            int(df.isna().sum().sum()),
        ]
    })

    outlier_summary = pd.DataFrame({
        "metric": [f"outliers_flagged_{col}" for col in outlier_counts],
        "value": list(outlier_counts.values())
    })
    summary = pd.concat([summary, outlier_summary], ignore_index=True)
    summary.to_csv(SUMMARY_FILE, index=False)

    print("\nCleaning completed!")
    print(f"Cleaned dataset: {OUTPUT_FILE}")
    print(f"Quality summary: {SUMMARY_FILE}")
    print(f"Rows before cleaning: {original_rows}")
    print(f"Rows after cleaning: {len(df)}")
    print(f"Duplicate rows removed: {duplicates_before}")
    print("\nFinal dataset shape:", df.shape)
    print("\nFiles saved successfully.")

if __name__ == "__main__":
    main()
