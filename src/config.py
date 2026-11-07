# ============================================================
# STOCK MARKET ANALYTICS AND PORTFOLIO OPTIMISATION
# File: config.py
# Purpose: Store common project settings in one place
# ============================================================

# Import Path to manage file and folder locations
from pathlib import Path


# ------------------------------------------------------------
# 1. PROJECT FOLDER LOCATIONS
# ------------------------------------------------------------

# Find the location of the current config.py file
CURRENT_FILE = Path(__file__).resolve()

# Move up two folders to reach the main project directory
PROJECT_ROOT = CURRENT_FILE.parent.parent

# Define the main data folder
DATA_DIR = PROJECT_ROOT / "data"

# Define the folder for original downloaded stock data
RAW_DATA_DIR = DATA_DIR / "raw"

# Define the folder for cleaned stock data
PROCESSED_DATA_DIR = DATA_DIR / "processed"


# ------------------------------------------------------------
# 2. STOCK UNIVERSE
# ------------------------------------------------------------

# Store our 20 selected NSE companies and their sectors
# The stock ticker is the dictionary key
# Company name and sector are stored as dictionary values

STOCKS = {

    "TCS.NS": {"company": "Tata Consultancy Services", "sector": "IT"},
    "INFY.NS": {"company": "Infosys", "sector": "IT"},
    "HCLTECH.NS": {"company": "HCL Technologies", "sector": "IT"},

    "HDFCBANK.NS": {"company": "HDFC Bank", "sector": "Banking"},
    "ICICIBANK.NS": {"company": "ICICI Bank", "sector": "Banking"},
    "SBIN.NS": {"company": "State Bank of India", "sector": "Banking"},

    "ITC.NS": {"company": "ITC", "sector": "FMCG"},
    "HINDUNILVR.NS": {"company": "Hindustan Unilever", "sector": "FMCG"},
    "NESTLEIND.NS": {"company": "Nestle India", "sector": "FMCG"},

    "SUNPHARMA.NS": {"company": "Sun Pharma", "sector": "Pharma"},
    "CIPLA.NS": {"company": "Cipla", "sector": "Pharma"},

    "RELIANCE.NS": {"company": "Reliance Industries", "sector": "Energy"},
    "ONGC.NS": {"company": "ONGC", "sector": "Energy"},

    "NTPC.NS": {"company": "NTPC", "sector": "Utilities"},

    "MARUTI.NS": {"company": "Maruti Suzuki", "sector": "Automobile"},
    "BAJAJ-AUTO.NS": {"company": "Bajaj Auto", "sector": "Automobile"},

    "LT.NS": {"company": "Larsen and Toubro", "sector": "Industrial"},

    "BHARTIARTL.NS": {"company": "Bharti Airtel", "sector": "Telecom"},

    "BAJFINANCE.NS": {"company": "Bajaj Finance", "sector": "Financial Services"},

    "TITAN.NS": {"company": "Titan Company", "sector": "Consumer"}
}


# Extract all 20 ticker symbols into a separate list
STOCK_TICKERS = list(STOCKS.keys())


# ------------------------------------------------------------
# 3. MARKET BENCHMARK
# ------------------------------------------------------------

# Yahoo Finance ticker for the Nifty 50 index
BENCHMARK_TICKER = "^NSEI"


# ------------------------------------------------------------
# 4. HISTORICAL DATA SETTINGS
# ------------------------------------------------------------

# Request five years of historical stock data
HISTORY_PERIOD = "5y"

# Request daily stock market prices
DATA_INTERVAL = "1d"

# Keep original OHLC prices and adjusted closing prices.
# We will use adjusted prices when calculating investment returns.
AUTO_ADJUST = False