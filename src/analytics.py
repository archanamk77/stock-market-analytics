
# ============================================================
# STOCK MARKET ANALYTICS AND PORTFOLIO OPTIMISATION
# File: analytics.py
# Purpose: Calculate financial performance metrics for 20 stocks
# ============================================================

# Import pandas for financial data manipulation
import pandas as pd

# Import NumPy for mathematical calculations
import numpy as np

# Import Path for working with project folders
from pathlib import Path

# Import the configured stock universe and processed data path
from src.config import STOCKS, PROCESSED_DATA_DIR


# ============================================================
# 1. DEFINE PROJECT PATHS AND FINANCIAL SETTINGS
# ============================================================

# Locate the folder containing accepted cleaned datasets
ACCEPTED_DATA_DIR = PROCESSED_DATA_DIR / "batch" / "accepted"

# Create a folder for financial analysis outputs
ANALYTICS_DIR = PROCESSED_DATA_DIR / "analytics"

# Use 252 trading sessions as the annualisation convention
TRADING_DAYS = 252


# ============================================================
# 2. CONVERT TICKERS INTO SAFE FILENAMES
# ============================================================

def safe_ticker_name(ticker: str) -> str:

    # Convert the ticker to uppercase and replace special characters
    return (
        ticker.upper()
        .replace("^", "")
        .replace(".", "_")
        .replace("-", "_")
    )


# ============================================================
# 3. LOAD ONE CLEANED STOCK DATASET
# ============================================================

def load_stock_data(ticker: str) -> pd.DataFrame:

    # Generate the filename used by the cleaning module
    safe_ticker = safe_ticker_name(ticker)

    # Construct the cleaned CSV file location
    file_path = ACCEPTED_DATA_DIR / f"{safe_ticker}_clean.csv"

    # Check that the cleaned dataset exists
    if not file_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found for {ticker}: {file_path}"
        )

    # Load the cleaned historical stock data
    data = pd.read_csv(file_path, parse_dates=["Date"])

    # Define the columns required for financial analysis
    required_columns = ["Date", "Adj Close", "Close"]

    # Identify missing required columns
    missing = [
        column for column in required_columns
        if column not in data.columns
    ]

    # Stop if the dataset does not contain required fields
    if missing:
        raise ValueError(
            f"{ticker} is missing required columns: {missing}"
        )

    # Sort the records chronologically
    data = data.sort_values("Date").copy()

    # Convert adjusted closing prices into numeric values
    data["Adj Close"] = pd.to_numeric(
        data["Adj Close"],
        errors="coerce"
    )

    # Remove records without a usable date or adjusted closing price
    data = data.dropna(subset=["Date", "Adj Close"])

    # Reject duplicate dates instead of silently keeping one
    if data["Date"].duplicated().any():
        raise ValueError(f"Duplicate dates found for {ticker}")

    # Reject non-positive adjusted closing prices
    if (data["Adj Close"] <= 0).any():
        raise ValueError(
            f"Non-positive adjusted closing prices found for {ticker}"
        )

    # Set the trading date as the DataFrame index
    data = data.set_index("Date")

    # Give the adjusted closing price column its stock ticker name
    return data["Adj Close"].rename(ticker)


# ============================================================
# 4. LOAD ALL 20 STOCKS INTO ONE PRICE DATAFRAME
# ============================================================

def load_all_stock_prices() -> pd.DataFrame:

    # Create a dictionary to store individual stock price series
    price_series = {}

    # Process each ticker in our configured stock universe
    for ticker in STOCKS:

        # Load the cleaned adjusted closing price series
        prices = load_stock_data(ticker)

        # Store the series using its ticker as the key
        price_series[ticker] = prices

        # Display the loading status
        print(f"Loaded {ticker}: {len(prices)} observations")

    # Combine all stock prices using their trading dates
    # Outer alignment preserves dates even if one stock is missing a session
    prices_df = pd.concat(
        price_series.values(),
        axis=1,
        join="outer"
    ).sort_index()

    # Verify that all 20 configured stocks were loaded
    if prices_df.shape[1] != len(STOCKS):
        raise ValueError("The loaded stock count does not match config.py")

    # Return the combined adjusted closing prices
    return prices_df


# ============================================================
# 5. CALCULATE DAILY STOCK RETURNS
# ============================================================

def calculate_daily_returns(
    prices: pd.DataFrame
) -> pd.DataFrame:

    # Calculate percentage change between consecutive available sessions
    # fill_method=None prevents artificial price filling
    daily_returns = prices.pct_change(fill_method=None)

    # Return the daily percentage returns
    return daily_returns


# ============================================================
# 6. CALCULATE CUMULATIVE RETURNS
# ============================================================

def calculate_cumulative_returns(
    prices: pd.DataFrame
) -> pd.DataFrame:

    # Identify the first available adjusted closing price for each stock
    initial_prices = prices.apply(
        lambda column: column.dropna().iloc[0]
        if column.notna().any()
        else np.nan
    )

    # Calculate cumulative growth relative to the initial observation
    cumulative_returns = prices.divide(initial_prices).subtract(1)

    # Return cumulative percentage returns as decimal values
    return cumulative_returns


# ============================================================
# 7. CALCULATE FINANCIAL PERFORMANCE METRICS
# ============================================================

def calculate_performance_metrics(
    prices: pd.DataFrame,
    daily_returns: pd.DataFrame,
    risk_free_rate_annual: float | None = None
) -> pd.DataFrame:

    # Create an empty list to hold company-level metrics
    metrics = []

    # Process each stock separately
    for ticker in prices.columns:

        # Retrieve the adjusted closing price history
        stock_prices = prices[ticker].dropna()

        # Retrieve valid daily returns
        stock_returns = daily_returns[ticker].dropna()

        # Skip stocks with insufficient historical observations
        if len(stock_prices) < 2 or len(stock_returns) < 2:
            continue

        # Retrieve company and sector information
        company_info = STOCKS[ticker]

        # Calculate the total holding-period return
        total_return = (
            stock_prices.iloc[-1] / stock_prices.iloc[0]
        ) - 1

        # Calculate the elapsed calendar years
        elapsed_years = (
            stock_prices.index[-1] - stock_prices.index[0]
        ).days / 365.25

        # Calculate annualised compounded return
        annualised_return = (
            (1 + total_return) ** (1 / elapsed_years) - 1
            if elapsed_years > 0
            else np.nan
        )

        # Calculate annualised historical volatility
        annualised_volatility = (
            stock_returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
        )

        # Calculate the running highest adjusted closing price
        running_peak = stock_prices.cummax()

        # Calculate drawdown relative to the running peak
        drawdown = (stock_prices / running_peak) - 1

        # Find the maximum drawdown during the observation period
        max_drawdown = drawdown.min()

        # Calculate Sharpe ratio only when a risk-free rate is supplied
        if risk_free_rate_annual is not None:

            # Validate the annual risk-free rate
            if risk_free_rate_annual <= -1:
                raise ValueError("Risk-free rate must be greater than -100%")

            # Convert the annual risk-free rate into a daily rate
            daily_risk_free_rate = (
                (1 + risk_free_rate_annual)
                ** (1 / TRADING_DAYS) - 1
            )

            # Calculate excess daily returns
            excess_returns = stock_returns - daily_risk_free_rate

            # Calculate Sharpe ratio
            if stock_returns.std(ddof=1) > 0:
                sharpe_ratio = (
                    excess_returns.mean() /
                    stock_returns.std(ddof=1)
                ) * np.sqrt(TRADING_DAYS)
            else:
                sharpe_ratio = np.nan

        else:

            # Leave Sharpe ratio unavailable until configured
            sharpe_ratio = np.nan

        # Store all financial metrics for the stock
        metrics.append({

            # Store the stock ticker
            "Ticker": ticker,

            # Store the company name
            "Company": company_info["company"],

            # Store the business sector
            "Sector": company_info["sector"],

            # Store the first available trading date
            "First Date": stock_prices.index[0],

            # Store the latest available trading date
            "Last Date": stock_prices.index[-1],

            # Store the number of valid daily returns
            "Observations": len(stock_returns),

            # Store holding-period return as a percentage
            "Total Return (%)": total_return * 100,

            # Store annualised return as a percentage
            "Annualised Return (%)": annualised_return * 100,

            # Store annualised volatility as a percentage
            "Annualised Volatility (%)": annualised_volatility * 100,

            # Store maximum drawdown as a percentage
            "Maximum Drawdown (%)": max_drawdown * 100,

            # Store Sharpe ratio when a risk-free rate is configured
            "Sharpe Ratio": sharpe_ratio
        })

    # Convert the collected results into a DataFrame
    metrics_df = pd.DataFrame(metrics)

    # Sort companies by annualised return
    metrics_df = metrics_df.sort_values(
        "Annualised Return (%)",
        ascending=False
    ).reset_index(drop=True)

    # Return the consolidated performance report
    return metrics_df


# ============================================================
# 8. SAVE ALL FINANCIAL ANALYSIS OUTPUTS
# ============================================================

def save_analysis_outputs(
    prices: pd.DataFrame,
    daily_returns: pd.DataFrame,
    cumulative_returns: pd.DataFrame,
    metrics: pd.DataFrame
) -> None:

    # Create the analytics output folder
    ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)

    # Save adjusted closing prices
    prices.to_csv(
        ANALYTICS_DIR / "adjusted_closing_prices.csv"
    )

    # Save calculated daily returns
    daily_returns.to_csv(
        ANALYTICS_DIR / "daily_returns.csv"
    )

    # Save cumulative returns
    cumulative_returns.to_csv(
        ANALYTICS_DIR / "cumulative_returns.csv"
    )

    # Save company-level financial metrics
    metrics.to_csv(
        ANALYTICS_DIR / "performance_metrics.csv",
        index=False
    )

    # Confirm the output folder
    print("\nFINANCIAL ANALYSIS FILES SAVED")
    print("Output folder:", ANALYTICS_DIR)


# ============================================================
# 9. RUN THE COMPLETE FINANCIAL ANALYSIS
# ============================================================

def run_financial_analysis():

    # Display the beginning of the analysis
    print("=" * 60)
    print("STARTING 20-STOCK FINANCIAL ANALYSIS")
    print("=" * 60)

    # Load adjusted closing prices for all stocks
    prices = load_all_stock_prices()

    # Calculate daily returns
    daily_returns = calculate_daily_returns(prices)

    # Calculate cumulative returns
    cumulative_returns = calculate_cumulative_returns(prices)


    # Calculate performance metrics using an assumed annual risk-free rate of 6%
    metrics = calculate_performance_metrics(
        prices,
        daily_returns,
        risk_free_rate_annual=0.06
    )


    # Save all analysis outputs
    save_analysis_outputs(
        prices,
        daily_returns,
        cumulative_returns,
        metrics
    )

    # Display the financial performance report
    print("\nSTOCK PERFORMANCE SUMMARY")

    print(
        metrics[
            [
                "Ticker",
                "Total Return (%)",
                "Annualised Return (%)",
                "Annualised Volatility (%)",
                "Maximum Drawdown (%)"
            ]
        ].round(2).to_string(index=False)
    )

    # Display the number of stocks successfully analysed
    print("\nStocks analysed:", len(metrics))

    # Return all the generated DataFrames
    return prices, daily_returns, cumulative_returns, metrics


# ============================================================
# 10. ALLOW DIRECT EXECUTION FROM THE TERMINAL
# ============================================================

if __name__ == "__main__":

    # Start the financial analysis
    run_financial_analysis()
