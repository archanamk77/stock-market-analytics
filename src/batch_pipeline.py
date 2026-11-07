
# ============================================================
# STOCK MARKET ANALYTICS AND PORTFOLIO OPTIMISATION
# File: batch_pipeline.py
# Purpose: Collect and clean historical data for 20 stocks
# ============================================================

# Import time for a short pause between downloads
import time

# Import pandas for reading and processing CSV files
import pandas as pd

# Import Path for handling file locations
from pathlib import Path

# Import stock universe and folder locations
from src.config import STOCKS, RAW_DATA_DIR, PROCESSED_DATA_DIR

# Import existing data collection functions
from src.data_collection import download_stock_data, save_stock_data

# Import reusable cleaning functions
from src.data_cleaning import clean_stock_data, save_cleaning_outputs


# ============================================================
# 1. DEFINE FILE NAMING LOGIC
# ============================================================

def safe_ticker_name(ticker: str) -> str:

    # Convert the ticker into a filename-safe format
    return (
        ticker.upper()
        .replace("^", "")
        .replace(".", "_")
        .replace("-", "_")
    )


# ============================================================
# 2. LOAD EXISTING DATA OR DOWNLOAD MISSING DATA
# ============================================================

def load_stock_dataset(ticker: str) -> tuple[pd.DataFrame, str]:

    # Convert the ticker into a safe filename
    safe_ticker = safe_ticker_name(ticker)

    # Define the original raw CSV location
    raw_path = RAW_DATA_DIR / f"{safe_ticker}.csv"

    # Define the repaired TCS dataset location
    tcs_repaired_path = (
        PROCESSED_DATA_DIR / "TCS_NS_repaired.csv"
    )

    # Prefer our repaired TCS dataset over the original raw copy
    if ticker == "TCS.NS" and tcs_repaired_path.exists():

        # Read the repaired TCS historical data
        data = pd.read_csv(
            tcs_repaired_path,
            parse_dates=["Date"]
        )

        return data, str(tcs_repaired_path)

    # Reuse an existing raw dataset rather than overwriting it
    if raw_path.exists():

        # Read the original downloaded stock data
        data = pd.read_csv(
            raw_path,
            parse_dates=["Date"]
        )

        return data, str(raw_path)

    # Download historical prices when no raw CSV exists
    data = download_stock_data(ticker)

    # Verify that the download contains records
    if data is None or data.empty:
        raise ValueError(f"No historical data received for {ticker}")

    # Save the original downloaded dataset
    saved_path = save_stock_data(data, ticker)

    # Reload the saved file to standardise its format
    data = pd.read_csv(
        saved_path,
        parse_dates=["Date"]
    )

    return data, str(saved_path)


# ============================================================
# 3. DEFINE DOCUMENTED STOCK-SPECIFIC CLEANING RULES
# ============================================================


# ============================================================
# 3. DEFINE NSE-VERIFIED STOCK-SPECIFIC CLEANING RULES
# ============================================================

def get_cleaning_rules(
    ticker: str,
    data: pd.DataFrame
) -> tuple[dict, dict]:

    # Initialise correction and exclusion dictionaries
    corrections = {}
    exclusions = {}

    # Work on a separate copy
    data = data.copy()

    # Standardise the date column
    data["Date"] = pd.to_datetime(data["Date"])

    # Locate the folder containing official NSE references
    reference_dir = RAW_DATA_DIR / "reference"

    # Define the March 2025 official reference file
    march_path = (
        reference_dir / "NSE_EQ_Bhavcopy_20250318.csv"
    )

    # Stop if the official reference is unavailable
    if not march_path.exists():
        raise FileNotFoundError(
            f"NSE March reference missing: {march_path}"
        )

    # Load official March NSE data
    march_reference = pd.read_csv(march_path)

    # Standardise NSE ticker symbols
    march_reference["TckrSymb"] = (
        march_reference["TckrSymb"].astype(str).str.strip().str.upper()
    )

    # Keep only regular equity-series observations
    march_reference = march_reference[
        march_reference["SctySrs"].astype(str).str.strip() == "EQ"
    ]

    # Define the official NSE price mappings
    price_mapping = {
        "Open": "OpnPric",
        "High": "HghPric",
        "Low": "LwPric",
        "Close": "ClsPric"
    }


    # --------------------------------------------------------
    # HELPER: RECONSTRUCT ADJUSTED CLOSE
    # --------------------------------------------------------

    def calculate_adjusted_close(
        target_date: str,
        official_close: float
    ) -> float:

        # Convert the target date to a timestamp
        target_timestamp = pd.Timestamp(target_date)

        # Retrieve the previous trading observation
        previous = data[
            data["Date"] < target_timestamp
        ].sort_values("Date").tail(1)

        # Retrieve the next trading observation
        following = data[
            data["Date"] > target_timestamp
        ].sort_values("Date").head(1)

        # Ensure both neighbouring observations exist
        if len(previous) != 1 or len(following) != 1:
            raise ValueError(
                f"Cannot reconstruct adjusted close for {ticker} "
                f"on {target_date}."
            )

        # Calculate the previous adjustment factor
        previous_close = float(previous["Close"].iloc[0])
        previous_adjusted = float(previous["Adj Close"].iloc[0])

        # Calculate the following adjustment factor
        following_close = float(following["Close"].iloc[0])
        following_adjusted = float(following["Adj Close"].iloc[0])

        # Reject invalid price observations
        if min(previous_close, following_close) <= 0:
            raise ValueError("Invalid neighbouring closing prices.")

        if previous_adjusted <= 0 or following_adjusted <= 0:
            raise ValueError("Invalid neighbouring adjusted prices.")

        # Derive both historical adjustment factors
        factor_before = previous_adjusted / previous_close
        factor_after = following_adjusted / following_close

        # Verify consistency before reconstruction
        if abs(factor_before - factor_after) > 0.000001:
            raise ValueError(
                f"Adjustment factors inconsistent for {ticker} "
                f"on {target_date}. Manual investigation required."
            )

        # Calculate the average adjustment factor
        average_factor = (factor_before + factor_after) / 2

        # Calculate the adjusted closing price
        return official_close * average_factor


    # --------------------------------------------------------
    # HELPER: BUILD A VERIFIED NSE CORRECTION
    # --------------------------------------------------------

    def build_verified_correction(
        target_date: str,
        official_record: pd.Series,
        source: str,
        source_url: str,
        reason: str
    ) -> dict | None:

        # Locate the original stock observation
        target_timestamp = pd.Timestamp(target_date)

        matching_rows = data[
            data["Date"] == target_timestamp
        ]

        # Check that exactly one record exists
        if len(matching_rows) != 1:
            raise ValueError(
                f"Expected exactly one record for {ticker} "
                f"on {target_date}; found {len(matching_rows)}."
            )

        # Retrieve the original Yahoo Finance record
        original = matching_rows.iloc[0]

        # Extract official NSE OHLC prices

        # Corporate action adjustment factors
        # These convert historical NSE prices into Yahoo's current share basis
        corporate_action_factors = {
            "BAJFINANCE.NS": 10.0,
            "HDFCBANK.NS": 2.0,
            "NESTLEIND.NS": 2.0
        }

        # Get the correct factor for the current stock
        # Stocks without these corporate actions use factor 1
        adjustment_factor = corporate_action_factors.get(ticker, 1.0)

        # Retrieve NSE prices and convert them to the adjusted share basis
        official_values = {
            column: (
                float(official_record[nse_column]) / adjustment_factor
            )
            for column, nse_column in price_mapping.items()
        }

        # Adjust trading volume to account for the increased share count
        official_volume = int(round(
            float(official_record["TtlTradgVol"]) * adjustment_factor
        ))

        # Display the adjustment applied
        print(
            f"{ticker}: Corporate action factor = {adjustment_factor}"
        )


        # Determine whether any OHLC price is incorrect
        price_mismatch = any(
            pd.isna(original[column]) or
            abs(float(original[column]) - official_values[column]) > 0.02
            for column in price_mapping
        )

        # Determine whether trading volume is incorrect
        volume_mismatch = (
            pd.isna(original["Volume"]) or
            float(original["Volume"]) != official_volume
        )

        # Skip corrections when Yahoo already matches NSE
        if not price_mismatch and not volume_mismatch:
            return None

        # Reconstruct adjusted close using neighbouring sessions
        estimated_adjusted_close = calculate_adjusted_close(
            target_date,
            official_values["Close"]
        )

        # Build the complete replacement values
        new_values = {
            **official_values,
            "Adj Close": estimated_adjusted_close,
            "Volume": official_volume
        }

        # Return a correction with its supporting evidence
        return {
            "values": new_values,
            "reason": reason,
            "source": source,
            "source_url": source_url
        }


    # ========================================================
    # 4. CORRECT MARCH 18, 2025 USING OFFICIAL NSE DATA
    # ========================================================

    # Convert Yahoo Finance ticker into NSE symbol
    symbol = ticker.removesuffix(".NS").upper()

    # Find the corresponding official NSE equity observation
    march_record = march_reference[
        march_reference["TckrSymb"] == symbol
    ]

    # Require one official NSE record
    if len(march_record) != 1:
        raise ValueError(
            f"Expected one NSE EQ reference for {symbol}: "
            f"found {len(march_record)}."
        )
    # Retrieve the exchange record
    march_record = march_record.iloc[0]

    # Construct the March correction if required
    march_correction = build_verified_correction(
        target_date="2025-03-18",
        official_record=march_record,
        source="NSE Official CM-UDiFF Bhavcopy — EQ Series",
        source_url=(
            "https://nsearchives.nseindia.com/content/cm/"
            "BhavCopy_NSE_CM_0_0_0_20250318_F_0000.csv.zip"
        ),
        reason=(
            "Yahoo Finance zero-volume/stale-price anomaly. "
            "OHLC and volume verified against official NSE EQ data."
        )
    )

    # Record the March correction when a mismatch is present
    if march_correction is not None:
        corrections["2025-03-18"] = march_correction


    # ========================================================
    # 5. CORRECT BAJAJ AUTO — JANUARY 15, 2024
    # ========================================================

    # Apply this additional reference only to Bajaj Auto
    if ticker == "BAJAJ-AUTO.NS":

        # Define the official January reference file
        january_path = (
            reference_dir / "BAJAJ_AUTO_NSE_20240115.csv"
        )

        # Stop if the official reference is missing
        if not january_path.exists():
            raise FileNotFoundError(
                f"Bajaj Auto NSE reference missing: {january_path}"
            )

        # Load the official January NSE record
        january_reference = pd.read_csv(january_path)

        # Require an official equity-series record
        january_reference = january_reference[
            january_reference["SctySrs"].astype(str).str.strip() == "EQ"
        ]

        # Require exactly one official observation
        if len(january_reference) != 1:
            raise ValueError(
                "Bajaj Auto January NSE reference is ambiguous."
            )

        # Build the verified correction
        january_correction = build_verified_correction(
            target_date="2024-01-15",
            official_record=january_reference.iloc[0],
            source="NSE Official CM-UDiFF Bhavcopy — EQ Series",
            source_url=(
                "https://nsearchives.nseindia.com/content/cm/"
                "BhavCopy_NSE_CM_0_0_0_20240115_F_0000.csv.zip"
            ),
            reason=(
                "Yahoo Finance zero-volume/stale-price anomaly. "
                "OHLC and volume verified against official NSE EQ data."
            )
        )

        # Record the January correction when required
        if january_correction is not None:
            corrections["2024-01-15"] = january_correction


    # ========================================================
    # 6. EXCLUDE THE VERIFIED JANUARY 2026 NSE HOLIDAY
    # ========================================================

    # Check whether January 15, 2026 exists in the dataset
    holiday_date = pd.Timestamp("2026-01-15")

    if (data["Date"] == holiday_date).any():

        # Record the officially verified trading holiday
        exclusions["2026-01-15"] = {
            "reason": "Official NSE equity trading holiday.",
            "source": "NSE 2026 Equity Holiday Calendar",
            "source_url": (
                "https://www.nseindia.com/resources/"
                "exchange-communication-holidays?apiversion=v2"
            )
        }

    # Return all verified corrections and exclusions
    return corrections, exclusions



# ============================================================
# 4. RUN THE COMPLETE 20-STOCK PIPELINE
# ============================================================

def run_batch_pipeline():

    # Create a separate batch-processing output folder
    batch_dir = PROCESSED_DATA_DIR / "batch"

    # Create the folder if necessary
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Store successful data-quality reports
    quality_results = []

    # Store any stocks that could not be processed
    failed_stocks = []

    # Calculate how many stocks are in our universe
    total_stocks = len(STOCKS)

    # Display the beginning of batch processing
    print("=" * 60)
    print("STARTING 20-STOCK DATA PIPELINE")
    print("=" * 60)

    # Process each stock individually
    for number, (ticker, details) in enumerate(
        STOCKS.items(), start=1
    ):

        # Extract the company name
        company = details["company"]

        # Extract the sector
        sector = details["sector"]

        # Display the current processing status
        print(f"\n[{number}/{total_stocks}] {company} ({ticker})")

        try:

            # Load existing historical data or download it
            stock_data, source_path = load_stock_dataset(ticker)

            # Display the data source and record count
            print("Source:", source_path)
            print("Downloaded records:", len(stock_data))

            # Generate documented corrections and exclusions
            corrections, exclusions = get_cleaning_rules(
                ticker,
                stock_data
            )

            # Run the reusable cleaning function
            cleaned_data, audit_log, report = clean_stock_data(
                data=stock_data,
                ticker=ticker,
                corrections=corrections,
                exclusions=exclusions
            )

            # Identify issues requiring human investigation
            review_required = (
                report["Missing Price Values"] > 0
                or report["Duplicate Dates"] > 0
                or report["Zero Volume Records"] > 0
                or report["Negative Volume Records"] > 0
                or report["Invalid OHLC Records"] > 0
                or not report["Chronological Order"]
            )

            # Decide the destination based on quality results
            status_folder = (
                "review_required"
                if review_required
                else "accepted"
            )

            # Create the stock-specific output directory
            stock_output_dir = batch_dir / status_folder

            # Save the cleaned data and its audit log
            clean_path, audit_path = save_cleaning_outputs(
                cleaned_data=cleaned_data,
                audit_log=audit_log,
                ticker=ticker,
                output_dir=stock_output_dir
            )

            # Add company details to the quality report
            report["Company"] = company
            report["Sector"] = sector
            report["Source File"] = source_path
            report["Status"] = (
                "REVIEW REQUIRED"
                if review_required
                else "PASSED"
            )

            # Record the output locations
            report["Cleaned File"] = str(clean_path)
            report["Audit File"] = str(audit_path)

            # Store the completed quality report
            quality_results.append(report)

            # Display individual processing results
            print("Cleaned records:", len(cleaned_data))
            print("Status:", report["Status"])
            print("Audit entries:", len(audit_log))

        except Exception as error:

            # Record the failure without stopping other stocks
            failed_stocks.append({
                "Ticker": ticker,
                "Company": company,
                "Sector": sector,
                "Error": str(error)
            })

            # Display the error for investigation
            print("PROCESSING FAILED:", error)

        # Pause briefly between stock downloads
        time.sleep(0.5)

    # ========================================================
    # 5. GENERATE CONSOLIDATED QUALITY REPORT
    # ========================================================

    # Convert all successful quality reports into a DataFrame
    quality_df = pd.DataFrame(quality_results)

    # Convert failed stock information into a DataFrame
    failed_df = pd.DataFrame(failed_stocks)

    # Define the consolidated quality report location
    quality_path = batch_dir / "batch_quality_report.csv"

    # Define the failed stock report location
    failed_path = batch_dir / "failed_stocks_report.csv"

    # Save the successful quality reports
    quality_df.to_csv(quality_path, index=False)

    # Save the failure information
    failed_df.to_csv(failed_path, index=False)

    # Display the final batch summary
    print("\n" + "=" * 60)
    print("BATCH PIPELINE COMPLETED")
    print("=" * 60)

    print("Total stocks:", total_stocks)
    print("Successfully processed:", len(quality_results))
    print("Failed stocks:", len(failed_stocks))

    if not quality_df.empty:

        # Count stocks that passed all configured checks
        passed = (quality_df["Status"] == "PASSED").sum()

        # Count stocks requiring manual review
        review = (
            quality_df["Status"] == "REVIEW REQUIRED"
        ).sum()

        print("Passed:", passed)
        print("Require review:", review)

        # Display company-level quality results
        print("\nSTOCK-WISE QUALITY REPORT")

        print(
            quality_df[
                ["Ticker", "Company", "Total Records", "Status"]
            ].to_string(index=False)
        )

    # Display saved report locations
    print("\nQuality report:", quality_path)
    print("Failure report:", failed_path)

    # Return reports for further analysis
    return quality_df, failed_df


# ============================================================
# 6. ALLOW THE FILE TO RUN DIRECTLY FROM THE TERMINAL
# ============================================================

if __name__ == "__main__":

    # Start processing the complete stock universe
    run_batch_pipeline()
