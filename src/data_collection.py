
# ============================================================
# STOCK MARKET ANALYTICS AND PORTFOLIO OPTIMISATION
# File: data_collection.py
# Purpose: Download and save historical stock market data
# ============================================================


# Import yfinance to download historical stock market data
import yfinance as yf

# Import pandas to work with stock market data in table format
import pandas as pd

# Import Path to manage file and folder locations
from pathlib import Path

# Import our project settings from config.py
from src.config import (
    HISTORY_PERIOD,
    DATA_INTERVAL,
    AUTO_ADJUST,
    RAW_DATA_DIR
)


# ============================================================
# 1. DOWNLOAD STOCK DATA
# ============================================================

def download_stock_data(ticker: str) -> pd.DataFrame:
    """
    Download historical stock market data for one company.

    Parameters:
        ticker: Yahoo Finance stock symbol, e.g. TCS.NS

    Returns:
        A pandas DataFrame containing historical stock data.

    Raises:
        ValueError: If the ticker is invalid or no data is returned.
        RuntimeError: If the download fails.
    """

    # Check whether the supplied ticker is a valid non-empty string
    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("Please provide a valid stock ticker.")

    # Remove accidental spaces from the ticker symbol
    ticker = ticker.strip().upper()

    try:

        # Download historical stock data from Yahoo Finance
        data = yf.download(
            tickers=ticker,               # Stock symbol to download
            period=HISTORY_PERIOD,        # Download the last five years
            interval=DATA_INTERVAL,       # Retrieve daily data
            auto_adjust=AUTO_ADJUST,      # Preserve original OHLC prices
            actions=True,                 # Include dividends and stock splits
            progress=False,               # Hide the download progress bar
            threads=False,                # Avoid parallel requests for one ticker
            multi_level_index=False,      # Use normal column names
            timeout=30                    # Allow up to 30 seconds for the request
        )

        # Check whether Yahoo Finance returned any data
        if data is None or data.empty:
            raise ValueError(
                f"No historical data received for ticker: {ticker}"
            )

        # Convert the index into a proper datetime format
        data.index = pd.to_datetime(data.index)

        # Label the index as Date
        data.index.name = "Date"

        # Arrange the records chronologically
        data = data.sort_index()

        # Return the downloaded DataFrame
        return data

    except ValueError:
        # Pass validation errors to the calling program
        raise

    except Exception as error:
        # Convert unexpected download errors into a readable message
        raise RuntimeError(
            f"Failed to download data for {ticker}: {error}"
        ) from error


# ============================================================
# 2. SAVE STOCK DATA
# ============================================================

def save_stock_data(data: pd.DataFrame, ticker: str) -> Path:
    """
    Save downloaded stock data as a CSV file.

    Parameters:
        data: Historical stock data in DataFrame format.
        ticker: Yahoo Finance stock symbol.

    Returns:
        Path of the saved CSV file.
    """

    # Make sure the supplied DataFrame contains data
    if data is None or data.empty:
        raise ValueError("Cannot save an empty stock dataset.")

    # Create the raw data directory if it does not already exist
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Convert the ticker into a filename-safe format
    safe_ticker = (
        ticker.upper()
        .replace("^", "")
        .replace(".", "_")
        .replace("-", "_")
    )

    # Construct the CSV file path
    file_path = RAW_DATA_DIR / f"{safe_ticker}.csv"

    # Save the DataFrame, including its Date index
    data.to_csv(file_path, index=True)

    # Return the location of the saved file
    return file_path
