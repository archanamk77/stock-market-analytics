
# Import Path to locate project files reliably.
from pathlib import Path

# Import NumPy for numerical calculations.
import numpy as np

# Import Pandas for reading and manipulating CSV files.
import pandas as pd

# Import Streamlit to build the interactive dashboard.
import streamlit as st

# Import Matplotlib to create the correlation heatmap and efficient frontier.
import matplotlib.pyplot as plt


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

# Set the browser tab title and dashboard layout.
st.set_page_config(
    page_title="Stock Market Analytics",
    page_icon="📈",
    layout="wide"
)

# Find the project root directory (one level above app).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define the folder containing financial analytics CSV files.
ANALYTICS_DIR = PROJECT_ROOT / "data" / "processed" / "analytics"

# Define the folder containing portfolio optimisation outputs.
PORTFOLIO_DIR = PROJECT_ROOT / "data" / "processed" / "portfolio"

# Define the location of adjusted stock prices.
PRICE_FILE = ANALYTICS_DIR / "adjusted_closing_prices.csv"

# Define the location of daily stock returns.
RETURN_FILE = ANALYTICS_DIR / "daily_returns.csv"

# Define the location of cumulative stock returns.
CUMULATIVE_FILE = ANALYTICS_DIR / "cumulative_returns.csv"

# Define the location of stock performance metrics.
METRICS_FILE = ANALYTICS_DIR / "performance_metrics.csv"

# Define the location of the correlation matrix.
CORRELATION_FILE = PORTFOLIO_DIR / "correlation_matrix.csv"

# Define the location of optimised portfolio allocations.
WEIGHTS_FILE = PORTFOLIO_DIR / "optimal_portfolio_weights.csv"

# Define the location of optimised portfolio performance.
SUMMARY_FILE = PORTFOLIO_DIR / "portfolio_summary.csv"

# Define the location of the efficient frontier results.
FRONTIER_FILE = PORTFOLIO_DIR / "efficient_frontier.csv"


# ============================================================
# 2. DASHBOARD STYLING
# ============================================================

# Apply custom CSS to create a clean dark-themed dashboard.
st.markdown("""
<style>
    .stApp {
        background-color: #0b1120;
        color: #f1f5f9;
    }

    [data-testid="stMetric"] {
        background-color: #151f32;
        border: 1px solid #26344d;
        padding: 18px;
        border-radius: 12px;
    }

    [data-testid="stSidebar"] {
        background-color: #111a2b;
    }

    h1, h2, h3 {
        color: #f1f5f9;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. LOAD DATA FROM EXISTING CSV FILES
# ============================================================

# Cache CSV reading so the dashboard does not reload files unnecessarily.
@st.cache_data
def load_csv(file_path, index_col=None):

    # Read the specified CSV file into a Pandas DataFrame.
    return pd.read_csv(file_path, index_col=index_col)


# Cache historical price and return data in wide format.
@st.cache_data
def load_wide_csv(file_path):

    # Read the CSV file.
    data = pd.read_csv(file_path)

    # Confirm that a Date column is available.
    if "Date" not in data.columns:
        raise ValueError(f"Date column missing in {file_path}")

    # Convert the date column into datetime values.
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

    # Remove rows containing invalid dates.
    data = data.dropna(subset=["Date"])

    # Set Date as the DataFrame index.
    data = data.set_index("Date")

    # Sort rows in chronological order.
    data = data.sort_index()

    # Convert stock values to numeric values.
    data = data.apply(pd.to_numeric, errors="coerce")

    # Return the cleaned DataFrame.
    return data


# Define a helper function to load files safely.
def load_optional_file(file_path, wide=False, index_col=None):

    # Check whether the requested file exists.
    if not file_path.exists():

        # Return an empty DataFrame when a file is unavailable.
        return pd.DataFrame()

    # Load historical wide-format data when requested.
    if wide:
        return load_wide_csv(str(file_path))

    # Otherwise load a regular CSV file.
    return load_csv(str(file_path), index_col=index_col)


# ============================================================
# 4. LOAD ALL AVAILABLE PROJECT DATA
# ============================================================

# Load adjusted closing prices.
try:
    prices = load_optional_file(PRICE_FILE, wide=True)
except Exception as error:
    st.error(f"Unable to load adjusted prices: {error}")
    prices = pd.DataFrame()

# Load daily stock returns.
try:
    daily_returns = load_optional_file(RETURN_FILE, wide=True)
except Exception as error:
    st.error(f"Unable to load daily returns: {error}")
    daily_returns = pd.DataFrame()

# Load cumulative returns.
try:
    cumulative_returns = load_optional_file(CUMULATIVE_FILE, wide=True)
except Exception as error:
    st.warning(f"Unable to load cumulative returns: {error}")
    cumulative_returns = pd.DataFrame()

# Load individual stock performance metrics.
try:
    performance = load_optional_file(METRICS_FILE)
except Exception as error:
    st.warning(f"Unable to load performance metrics: {error}")
    performance = pd.DataFrame()

# Load the stock correlation matrix.
try:
    correlation = load_optional_file(CORRELATION_FILE, index_col=0)
except Exception as error:
    st.warning(f"Unable to load correlation matrix: {error}")
    correlation = pd.DataFrame()

# Load minimum-volatility and maximum-Sharpe allocations.
try:
    weights = load_optional_file(WEIGHTS_FILE)
except Exception as error:
    st.warning(f"Unable to load portfolio weights: {error}")
    weights = pd.DataFrame()

# Load optimised portfolio performance.
try:
    portfolio_summary = load_optional_file(SUMMARY_FILE)
except Exception as error:
    st.warning(f"Unable to load portfolio summary: {error}")
    portfolio_summary = pd.DataFrame()

# Load efficient frontier data.
try:
    frontier = load_optional_file(FRONTIER_FILE)
except Exception as error:
    st.warning(f"Unable to load efficient frontier: {error}")
    frontier = pd.DataFrame()


# ============================================================
# 5. VALIDATE REQUIRED DATA
# ============================================================

# Stop execution if the adjusted closing prices are unavailable.
if prices.empty:

    # Display the missing file location.
    st.error("Adjusted closing prices are unavailable.")

    # Provide the user with the expected file path.
    st.write(PRICE_FILE)

    # Stop rendering the dashboard.
    st.stop()


# ============================================================
# 6. DASHBOARD HEADER
# ============================================================

# Display the main application title.
st.title("📈 Stock Market Analytics & Portfolio Optimisation")

# Explain the purpose of the dashboard.
st.caption(
    "NSE Stock Analysis | Financial Performance | Risk Analytics | "
    "Modern Portfolio Theory"
)

# Display the historical period used for the dashboard.
st.caption(
    f"Historical data: {prices.index.min().date()} "
    f"to {prices.index.max().date()}"
)

# Display a divider below the header.
st.divider()

# Calculate complete trading observations for all stocks.
total_observations = (
    daily_returns.dropna(how="any").shape[0]
    if not daily_returns.empty
    else 0
)
# ============================================================
# 7. SIDEBAR NAVIGATION
# ============================================================

# Add a sidebar heading.
st.sidebar.title("Navigation")

# Create navigation options for the dashboard.
selected_page = st.sidebar.radio(
    "Select a section",
    [
        "Market Overview",
        "Stock Analysis",
        "Risk Analysis",
        "Portfolio Optimisation"
    ]
)

# Display the number of available stocks in the sidebar.
st.sidebar.metric("Stocks Available", prices.shape[1])

# Display the number of complete trading days.
st.sidebar.metric("Complete Trading Days", total_observations)
# Display the latest available market data date.
st.sidebar.caption(
    f"Latest data: {prices.index.max().date()}"
)


# ============================================================
# 8. MARKET OVERVIEW PAGE
# ============================================================

# Display the overview when the user selects Market Overview.
if selected_page == "Market Overview":

    # Display the page heading.
    st.header("Market Overview")

    # Calculate the total number of stocks.
    total_stocks = prices.shape[1]


    # Calculate average annualised volatility across all stocks.
    if not performance.empty and "Annualised Volatility (%)" in performance.columns:
        average_volatility = performance["Annualised Volatility (%)"].mean()
    else:
        average_volatility = 0.0

    # Calculate the average stock return when metrics are available.
    if "Total Return (%)" in performance.columns:
        average_return = performance["Total Return (%)"].mean()
    else:
        average_return = 0.0

    # Create four dashboard KPI columns.
    col1, col2, col3, col4 = st.columns(4)

    # Display the total number of stocks.
    col1.metric("Stocks Analysed", total_stocks)

    # Display the historical trading observations.
    col2.metric("Complete Trading Days", total_observations)

    # Display average annualised volatility.
    col3.metric(
        "Average Annualised Volatility",
        f"{average_volatility:.2f}%"
    )

    # Display the average cumulative return percentage.
    col4.metric("Average Total Return", f"{average_return:.2f}%")

    # Add spacing between dashboard sections.
    st.divider()

    # Display the stock performance heading.
    st.subheader("Stock Performance Overview")

    # Check whether stock performance metrics are available.
    if not performance.empty:

        # Display available stock performance metrics.
        st.dataframe(
            performance,
            use_container_width=True,
            hide_index=True
        )

        # Display a stock return comparison when possible.
        if "Total Return (%)" in performance.columns and "Ticker" in performance.columns:

            # Sort stocks by historical total return.
            top_stocks = performance.sort_values(
                "Total Return (%)",
                ascending=False
            ).head(10)

            # Display the top performing stocks.
            st.subheader("Top 10 Stocks by Total Return")

            # Plot stock returns using a bar chart.
            st.bar_chart(
                top_stocks.set_index("Ticker")[["Total Return (%)"]]
            )

    # Inform the user when the performance report is unavailable.
    else:
        st.info("Performance metrics CSV is unavailable.")

    # Display a historical market price chart.
    st.subheader("Historical Stock Prices")

    # Let the user select a stock to view.
    overview_ticker = st.selectbox(
        "Select stock",
        prices.columns,
        key="overview_ticker"
    )

    # Plot the selected stock's adjusted closing prices.
    st.line_chart(
        prices[[overview_ticker]].dropna()
    )


# ============================================================
# 9. INDIVIDUAL STOCK ANALYSIS PAGE
# ============================================================

# Display this section when Stock Analysis is selected.
if selected_page == "Stock Analysis":
    # Display the page title.
    st.header("Individual Stock Analysis")

    # Create a stock selector.
    ticker = st.selectbox(
        "Choose an NSE stock",
        prices.columns
    )

    # Extract adjusted closing prices for the selected stock.
    selected_prices = prices[ticker].dropna()

    # Create three columns for key stock metrics.
    col1, col2, col3 = st.columns(3)

    # Calculate the latest available stock price.
    latest_price = selected_prices.iloc[-1]

    # Calculate the earliest available stock price.
    initial_price = selected_prices.iloc[0]

    # Calculate historical total return.
    total_return = (
        (latest_price / initial_price) - 1
    ) * 100

    # Display the latest stock price.
    col1.metric("Latest Price", f"₹{latest_price:,.2f}")

    # Display the first historical stock price.
    col2.metric("Initial Price", f"₹{initial_price:,.2f}")

    # Display the historical total return.
    col3.metric("Total Return", f"{total_return:.2f}%")

    # Display the adjusted closing price chart.
    st.subheader("Adjusted Closing Price History")

    # Plot the selected stock's historical prices.
    st.line_chart(
        selected_prices.rename("Adjusted Closing Price")
    )

    # Check whether daily returns are available.
    if ticker in daily_returns.columns:

        # Display the daily return section.
        st.subheader("Daily Stock Returns (%)")

        # Convert decimal daily returns into percentage values.
        selected_daily_returns = (
            daily_returns[ticker].dropna() * 100
        )

        # Plot historical daily percentage returns.
        st.line_chart(
            selected_daily_returns.rename("Daily Return (%)")
        )

    # Check whether cumulative return data is available.
    if ticker in cumulative_returns.columns:

        # Display cumulative return analysis.
        st.subheader("Cumulative Returns (%)")

        # Convert decimal cumulative returns into percentages.
        selected_cumulative = (
            cumulative_returns[ticker].dropna() * 100
        )

        # Plot the selected stock's cumulative returns.
        st.line_chart(
            selected_cumulative.rename("Cumulative Return (%)")
        )

    # Display individual performance metrics when available.
    if not performance.empty and "Ticker" in performance.columns:

        # Find the performance record for the selected ticker.
        stock_metrics = performance[
            performance["Ticker"] == ticker
        ]

        # Display the matching performance record.
        if not stock_metrics.empty:

            # Add a heading for detailed metrics.
            st.subheader("Financial Performance Metrics")

            # Display the selected stock's metrics.
            st.dataframe(
                stock_metrics,
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# 10. RISK ANALYSIS PAGE
# ============================================================

# Display this page when Risk Analysis is selected.
elif selected_page == "Risk Analysis":

    # Display the page heading.
    st.header("Risk & Correlation Analysis")

    # Verify that performance metrics are available.
    if not performance.empty:

        # Identify important risk metrics.
        risk_columns = [
            "Ticker",
            "Annualised Volatility (%)",
            "Maximum Drawdown (%)",
            "Sharpe Ratio"
        ]

        # Keep only columns available in the performance report.
        available_risk_columns = [
            column for column in risk_columns
            if column in performance.columns
        ]

        # Display the risk comparison table.
        st.subheader("Risk Metrics by Stock")

        # Display risk information in a table.
        st.dataframe(
            performance[available_risk_columns],
            use_container_width=True,
            hide_index=True
        )

        # Plot annualised volatility when available.
        if "Annualised Volatility (%)" in performance.columns:

            # Sort stocks by volatility.
            volatility_data = performance.sort_values(
                "Annualised Volatility (%)",
                ascending=False
            ).set_index("Ticker")

            # Display a volatility chart.
            st.subheader("Annualised Stock Volatility")

            # Plot volatility for individual stocks.
            st.bar_chart(
                volatility_data[["Annualised Volatility (%)"]]
            )

    # Display the correlation matrix section.
    st.divider()
    st.subheader("Stock Correlation Heatmap")

    # Check whether the correlation matrix exists.
    if not correlation.empty:

        # Convert all correlation values into numeric values.
        correlation = correlation.apply(
            pd.to_numeric,
            errors="coerce"
        )

        # Create a Matplotlib figure for the heatmap.
        fig, ax = plt.subplots(figsize=(14, 10))

        # Display the correlation matrix as a heatmap.
        heatmap = ax.imshow(
            correlation.to_numpy(),
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            aspect="auto"
        )

        # Set the horizontal stock labels.
        ax.set_xticks(range(len(correlation.columns)))

        # Display ticker names along the horizontal axis.
        ax.set_xticklabels(
            correlation.columns,
            rotation=90,
            fontsize=7
        )

        # Set the vertical stock labels.
        ax.set_yticks(range(len(correlation.index)))

        # Display ticker names along the vertical axis.
        ax.set_yticklabels(
            correlation.index,
            fontsize=7
        )

        # Add a title to the heatmap.
        ax.set_title("NSE Stock Correlation Matrix")

        # Add a colour bar describing correlation strength.
        fig.colorbar(
            heatmap,
            ax=ax,
            label="Correlation"
        )

        # Adjust spacing around the figure.
        fig.tight_layout()

        # Display the heatmap inside Streamlit.
        st.pyplot(fig)

        # Close the Matplotlib figure.
        plt.close(fig)

        # Explain how to interpret correlation values.
        st.caption(
            "Correlation near +1 indicates similar movements; "
            "near -1 indicates opposite movements; "
            "near 0 indicates weaker linear relationships."
        )

    # Inform the user when correlation data is unavailable.
    else:
        st.warning("Correlation matrix CSV is unavailable.")


# ============================================================
# 11. PORTFOLIO OPTIMISATION PAGE
# ============================================================

# Display this section when Portfolio Optimisation is selected.
elif selected_page == "Portfolio Optimisation":

    # Display the page heading.
    st.header("Portfolio Optimisation")

    # Explain the methodology used for portfolio optimisation.
    st.caption(
        "Modern Portfolio Theory | Minimum Volatility | Maximum Sharpe"
    )

    # Check whether the portfolio summary is available.
    if not portfolio_summary.empty:

        # Display the portfolio comparison heading.
        st.subheader("Optimised Portfolio Performance")

        # Create one display column for each portfolio.
        col1, col2 = st.columns(2)

        # Find the minimum-volatility portfolio.
        minimum = portfolio_summary[
            portfolio_summary["Portfolio"] == "Minimum Volatility"
        ]

        # Find the maximum-Sharpe portfolio.
        maximum = portfolio_summary[
            portfolio_summary["Portfolio"] == "Maximum Sharpe"
        ]

        # Display the minimum-volatility portfolio metrics.
        if not minimum.empty:

            # Read the minimum-volatility result.
            row = minimum.iloc[0]

            # Show its annual expected return.
            col1.metric(
                "Minimum Risk: Expected Return",
                f"{row['Expected Annual Return (%)']:.2f}%"
            )

            # Show its annualised volatility.
            col1.metric(
                "Minimum Risk: Volatility",
                f"{row['Annualised Volatility (%)']:.2f}%"
            )

            # Show its Sharpe Ratio.
            col1.metric(
                "Minimum Risk: Sharpe Ratio",
                f"{row['Sharpe Ratio']:.3f}"
            )

        # Display the maximum-Sharpe portfolio metrics.
        if not maximum.empty:

            # Read the maximum-Sharpe result.
            row = maximum.iloc[0]

            # Show its annual expected return.
            col2.metric(
                "Maximum Sharpe: Expected Return",
                f"{row['Expected Annual Return (%)']:.2f}%"
            )

            # Show its annualised volatility.
            col2.metric(
                "Maximum Sharpe: Volatility",
                f"{row['Annualised Volatility (%)']:.2f}%"
            )

            # Show its Sharpe Ratio.
            col2.metric(
                "Maximum Sharpe: Sharpe Ratio",
                f"{row['Sharpe Ratio']:.3f}"
            )

        # Display both portfolio records for direct comparison.
        st.dataframe(
            portfolio_summary,
            use_container_width=True,
            hide_index=True
        )

    # Display a warning if the summary is unavailable.
    else:
        st.warning("Portfolio summary CSV is unavailable.")

    # Display the allocation analysis section.
    st.divider()
    st.subheader("Optimal Investment Allocations")

    # Check whether allocation results are available.
    if not weights.empty:

        # Let the user select an optimisation method.
        allocation_type = st.radio(
            "Choose portfolio",
            [
                "Maximum Sharpe",
                "Minimum Volatility"
            ],
            horizontal=True
        )

        # Identify the correct weight column.
        if allocation_type == "Maximum Sharpe":

            # Select weights from the maximum-Sharpe portfolio.
            weight_column = "Maximum Sharpe Weight (%)"

        else:

            # Select weights from the minimum-volatility portfolio.
            weight_column = "Minimum Volatility Weight (%)"

        # Verify that the requested allocation column exists.
        if weight_column in weights.columns:

            # Select ticker names and portfolio weights.
            allocation = weights[
                ["Ticker", weight_column]
            ].copy()

            # Set insignificant numerical values to zero.
            allocation.loc[
                allocation[weight_column].abs() < 1e-8,
                weight_column
            ] = 0.0

            # Sort allocations from largest to smallest.
            allocation = allocation.sort_values(
                weight_column,
                ascending=False
            )

            # Display the portfolio's total allocation.
            st.metric(
                "Total Portfolio Allocation",
                f"{allocation[weight_column].sum():.4f}%"
            )

            # Display the complete stock allocation table.
            st.dataframe(
                allocation.round(4),
                use_container_width=True,
                hide_index=True
            )

            # Display an allocation chart.
            st.subheader("Portfolio Weight Distribution")

            # Filter the allocations for the chart.
            chart_allocation = allocation[
                allocation[weight_column] > 0.01
            ]

            # Plot stocks with meaningful allocations.
            if not chart_allocation.empty:

                # Set ticker names as the chart index.
                chart_allocation = chart_allocation.set_index("Ticker")

                # Display the allocation bar chart.
                st.bar_chart(
                    chart_allocation[[weight_column]]
                )

            # Explain the absence of meaningful allocations.
            else:
                st.info("No allocations above the display threshold.")

    # Display a warning if allocation data is unavailable.
    else:
        st.warning("Optimal portfolio weights are unavailable.")

    # Display the efficient frontier section.
    st.divider()
    st.subheader("Efficient Frontier")

    # Verify that frontier observations are available.
    if not frontier.empty:

        # Create a Matplotlib figure.
        fig, ax = plt.subplots(figsize=(11, 7))

        # Sort the frontier observations by portfolio volatility.
        frontier_sorted = frontier.sort_values(
            "Annualised Volatility (%)"
        )

        # Plot the efficient frontier.
        ax.plot(
            frontier_sorted["Annualised Volatility (%)"],
            frontier_sorted["Expected Annual Return (%)"],
            marker="o",
            markersize=4,
            linewidth=2,
            label="Efficient Frontier"
        )

        # Add the minimum-volatility result if available.
        if not minimum.empty:

            # Mark the minimum-volatility portfolio.
            ax.scatter(
                minimum.iloc[0]["Annualised Volatility (%)"],
                minimum.iloc[0]["Expected Annual Return (%)"],
                marker="*",
                s=250,
                label="Minimum Volatility"
            )

        # Add the maximum-Sharpe result if available.
        if not maximum.empty:

            # Mark the maximum-Sharpe portfolio.
            ax.scatter(
                maximum.iloc[0]["Annualised Volatility (%)"],
                maximum.iloc[0]["Expected Annual Return (%)"],
                marker="*",
                s=250,
                label="Maximum Sharpe"
            )

        # Label the horizontal axis.
        ax.set_xlabel("Annualised Volatility (%)")

        # Label the vertical axis.
        ax.set_ylabel("Expected Annual Return (%)")

        # Add the efficient frontier title.
        ax.set_title("Risk vs Return: Efficient Frontier")

        # Display the legend.
        ax.legend()

        # Add a chart grid.
        ax.grid(alpha=0.3)

        # Adjust the chart layout.
        fig.tight_layout()

        # Display the chart inside Streamlit.
        st.pyplot(fig)

        # Close the figure.
        plt.close(fig)

        # Explain the efficient frontier.
        st.caption(
            "The efficient frontier represents portfolios offering "
            "the highest estimated return for a given level of risk."
        )

    # Inform the user when frontier observations are unavailable.
    else:
        st.warning("Efficient frontier CSV is unavailable.")


# ============================================================
# 12. APPLICATION FOOTER
# ============================================================

# Display a divider above the footer.
st.divider()

# Provide project information.
st.caption(
    "Stock Market Analytics & Portfolio Optimisation | "
    "Built with Python, Pandas, NumPy, SciPy, Matplotlib and Streamlit."
)

# Explain the historical nature of the portfolio estimates.
st.caption(
    "For educational and analytical purposes only. "
    "Historical returns do not guarantee future performance."
)
