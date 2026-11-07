
# ============================================================
# STOCK MARKET ANALYTICS AND PORTFOLIO OPTIMISATION
# File: data_cleaning.py
# Purpose: Validate, clean and audit historical stock data
# ============================================================

# Import pandas for data manipulation
import pandas as pd

# Import Path for handling file locations
from pathlib import Path

# Import JSON for storing old and new values in the audit log
import json

# Import project configuration
from src.config import PROCESSED_DATA_DIR


# ============================================================
# 1. DEFINE REQUIRED DATA COLUMNS
# ============================================================

# Define the columns required for financial analysis
REQUIRED_COLUMNS = [
    "Date", "Open", "High", "Low",
    "Close", "Adj Close", "Volume"
]

# Define columns that must contain numeric values
PRICE_COLUMNS = [
    "Open", "High", "Low", "Close", "Adj Close"
]

# Include trading volume in numeric validation
NUMERIC_COLUMNS = PRICE_COLUMNS + ["Volume"]


# ============================================================
# 2. CLEAN AND VALIDATE STOCK DATA
# ============================================================

def clean_stock_data(
    data: pd.DataFrame,
    ticker: str,
    corrections: dict | None = None,
    exclusions: dict | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Clean one stock's historical market dataset.

    Corrections and exclusions must be independently verified.
    The original input DataFrame is never modified.

    Returns:
        cleaned_data: Validated working dataset.
        audit_log: Record of all corrections and exclusions.
        quality_report: Summary of data-quality checks.
    """

    # Check that the input is a Pandas DataFrame
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a Pandas DataFrame.")

    # Check whether the input contains records
    if data.empty:
        raise ValueError("The supplied stock dataset is empty.")

    # Check whether the ticker is valid
    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("A valid stock ticker is required.")

    # Create a separate copy to protect the original dataset
    cleaned_data = data.copy()

    # Accept Date either as a column or as a named index
    if "Date" not in cleaned_data.columns:
        if cleaned_data.index.name == "Date":
            cleaned_data = cleaned_data.reset_index()
        else:
            raise ValueError("Dataset must contain a Date column.")

    # Check for all required financial columns
    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in cleaned_data.columns
    ]

    # Stop if important financial columns are missing
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Convert dates into Pandas datetime values
    cleaned_data["Date"] = pd.to_datetime(
        cleaned_data["Date"],
        errors="coerce"
    )

    # Reject records containing invalid dates
    if cleaned_data["Date"].isna().any():
        raise ValueError("Dataset contains invalid dates.")

    # Reject duplicate dates instead of silently deleting observations
    if cleaned_data["Date"].duplicated().any():
        raise ValueError("Duplicate trading dates detected.")

    # Convert financial columns to numeric values
    for column in NUMERIC_COLUMNS:
        cleaned_data[column] = pd.to_numeric(
            cleaned_data[column],
            errors="coerce"
        )

    # Initialise an empty list for audit records
    audit_entries = []

    # Use empty dictionaries when no corrections are supplied
    corrections = corrections or {}
    exclusions = exclusions or {}


    # ========================================================
    # 3. APPLY VERIFIED FINANCIAL CORRECTIONS
    # ========================================================

    # Process each explicitly supplied correction
    for date_string, details in corrections.items():

        # Convert the correction date into a timestamp
        correction_date = pd.Timestamp(date_string)

        # Find the corresponding trading record
        mask = cleaned_data["Date"] == correction_date

        # Ensure exactly one record exists for this date
        if mask.sum() != 1:
            raise ValueError(
                f"Expected one record for {date_string}."
            )

        # Extract the values that will be corrected
        updates = details["values"]

        # Reject correction columns that are not financial columns
        invalid_columns = set(updates) - set(NUMERIC_COLUMNS)

        if invalid_columns:
            raise ValueError(
                f"Invalid correction columns: {invalid_columns}"
            )

        # Preserve the original values before correction
        old_values = cleaned_data.loc[
            mask, list(updates)
        ].iloc[0].to_dict()

        # Validate and apply each corrected value
        for column, value in updates.items():

            # Convert the supplied corrected value to numeric
            numeric_value = pd.to_numeric(
                value,
                errors="raise"
            )

            # Apply the independently verified value
            cleaned_data.loc[mask, column] = numeric_value

        # Record the correction in the audit log
        audit_entries.append({
            "Ticker": ticker,
            "Date": date_string,
            "Action": "Corrected",
            "Reason": details["reason"],
            "Source": details["source"],
            "Source URL": details.get("source_url", ""),
            "Old Values": json.dumps(old_values, default=str),
            "New Values": json.dumps(updates, default=str)
        })


    # ========================================================
    # 4. EXCLUDE VERIFIED NON-TRADING DATES
    # ========================================================

    # Process each explicitly verified exclusion
    for date_string, details in exclusions.items():

        # Convert the exclusion date into a timestamp
        exclusion_date = pd.Timestamp(date_string)

        # Locate the corresponding record
        mask = cleaned_data["Date"] == exclusion_date

        # Ensure that the date exists exactly once
        if mask.sum() != 1:
            raise ValueError(
                f"Expected one record for {date_string}."
            )

        # Preserve the excluded record for audit purposes
        old_values = cleaned_data.loc[
            mask, REQUIRED_COLUMNS
        ].iloc[0].to_dict()

        # Add the exclusion to our audit log
        audit_entries.append({
            "Ticker": ticker,
            "Date": date_string,
            "Action": "Excluded",
            "Reason": details["reason"],
            "Source": details["source"],
            "Source URL": details.get("source_url", ""),
            "Old Values": json.dumps(old_values, default=str),
            "New Values": "{}"
        })

        # Remove the verified non-trading date
        cleaned_data = cleaned_data.loc[~mask].copy()


    # ========================================================
    # 5. SORT THE FINAL DATASET
    # ========================================================

    # Arrange stock records from oldest to newest
    cleaned_data = cleaned_data.sort_values("Date")

    # Reset the row index after exclusions
    cleaned_data = cleaned_data.reset_index(drop=True)


    # ========================================================
    # 6. PERFORM FINAL DATA QUALITY CHECKS
    # ========================================================

    # Extract the OHLC price columns
    prices = cleaned_data[PRICE_COLUMNS]

    # Extract trading volume
    volume = cleaned_data["Volume"]

    # Identify invalid OHLC price relationships
    invalid_ohlc = (
        (cleaned_data["High"] < cleaned_data["Low"]) |
        (cleaned_data["High"] < cleaned_data["Open"]) |
        (cleaned_data["High"] < cleaned_data["Close"]) |
        (cleaned_data["Low"] > cleaned_data["Open"]) |
        (cleaned_data["Low"] > cleaned_data["Close"])
    )

    # Prepare a dictionary containing final quality metrics
    quality_report = {
        "Ticker": ticker,
        "Total Records": len(cleaned_data),
        "Missing Closing Prices": int(
            cleaned_data["Close"].isna().sum()
        ),
        "Missing Price Values": int(
            prices.isna().sum().sum()
        ),
        "Duplicate Dates": int(
            cleaned_data["Date"].duplicated().sum()
        ),
        "Zero Volume Records": int(
            (volume == 0).sum()
        ),
        "Negative Volume Records": int(
            (volume < 0).sum()
        ),
        "Invalid OHLC Records": int(
            invalid_ohlc.sum()
        ),
        "Chronological Order": bool(
            cleaned_data["Date"].is_monotonic_increasing
        )
    }

    # Convert audit records into a DataFrame
    audit_log = pd.DataFrame(
        audit_entries,
        columns=[
            "Ticker", "Date", "Action", "Reason",
            "Source", "Source URL", "Old Values", "New Values"
        ]
    )

    # Return all three outputs
    return cleaned_data, audit_log, quality_report


# ============================================================
# 7. SAVE CLEANED DATA AND AUDIT LOG
# ============================================================

def save_cleaning_outputs(
    cleaned_data: pd.DataFrame,
    audit_log: pd.DataFrame,
    ticker: str,
    output_dir: Path | None = None
) -> tuple[Path, Path]:
    """
    Save cleaned historical data and its correction audit log.
    """

    # Validate the ticker
    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("A valid ticker is required.")

    # Use the project's processed folder by default
    if output_dir is None:
        output_dir = PROCESSED_DATA_DIR

    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate a safe filename from the ticker
    safe_ticker = (
        ticker.upper()
        .replace("^", "")
        .replace(".", "_")
        .replace("-", "_")
    )

    # Define the cleaned dataset file
    cleaned_path = output_dir / f"{safe_ticker}_clean.csv"

    # Define the audit log file
    audit_path = output_dir / f"{safe_ticker}_correction_log.csv"

    # Save the cleaned financial data
    cleaned_data.to_csv(cleaned_path, index=False)

    # Save the audit log
    audit_log.to_csv(audit_path, index=False)

    # Return both saved file locations
    return cleaned_path, audit_path
