
# Import Path to manage project folders and file locations.
from pathlib import Path

# Import NumPy for mathematical and numerical calculations.
import numpy as np

# Import Pandas to load and process stock return data.
import pandas as pd

# Import Matplotlib to generate the efficient frontier chart.
import matplotlib.pyplot as plt

# Import minimize to calculate optimal portfolio allocations.
from scipy.optimize import minimize


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

# Find the main project directory (one level above src).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define the folder containing the existing financial analytics.
ANALYTICS_DIR = PROJECT_ROOT / "data" / "processed" / "analytics"

# Define the folder where portfolio results will be saved.
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "portfolio"

# Define the location of the daily stock returns CSV.
RETURNS_FILE = ANALYTICS_DIR / "daily_returns.csv"

# Assume 252 trading days in one year.
TRADING_DAYS = 252

# Use the same illustrative annual risk-free rate as analytics.py.
RISK_FREE_RATE = 0.06

# Generate 10,000 randomly weighted portfolios.
NUMBER_OF_SIMULATIONS = 10_000

# Define the minimum historical observations required per stock.
MIN_OBSERVATIONS = 252

# Set a fixed random seed so simulations are reproducible.
RANDOM_SEED = 42


# ============================================================
# 2. LOAD AND PREPARE DAILY RETURNS
# ============================================================

def load_daily_returns():

    # Check whether the daily returns file exists.
    if not RETURNS_FILE.exists():
        raise FileNotFoundError(
            f"Daily returns file not found: {RETURNS_FILE}"
        )

    # Read the existing daily returns CSV into a DataFrame.
    data = pd.read_csv(RETURNS_FILE)

    # Verify that the dataset contains a Date column.
    if "Date" not in data.columns:
        raise ValueError("The daily returns file must contain Date.")

    # Convert the Date column into proper datetime values.
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

    # Remove rows where the trading date cannot be interpreted.
    data = data.dropna(subset=["Date"])

    # Set the trading date as the DataFrame index.
    data = data.set_index("Date")

    # Sort observations chronologically.
    data = data.sort_index()

    # Convert all stock return columns into numeric values.
    returns = data.apply(pd.to_numeric, errors="coerce")

    # Replace infinite values with missing values.
    returns = returns.replace([np.inf, -np.inf], np.nan)

    # Identify stocks with at least 252 valid observations.
    valid_columns = returns.columns[
        returns.notna().sum() >= MIN_OBSERVATIONS
    ]

    # Identify stocks excluded because of insufficient observations.
    excluded_columns = [
        column for column in returns.columns
        if column not in valid_columns
    ]

    # Keep only stocks with sufficient historical observations.
    returns = returns[valid_columns]

    # Remove dates where even one selected stock has a missing return.
    returns = returns.dropna(axis=0, how="any")

    # Check that at least two stocks are available.
    if returns.shape[1] < 2:
        raise ValueError("At least two valid stocks are required.")

    # Check that enough complete observations remain.
    if len(returns) < MIN_OBSERVATIONS:
        raise ValueError("Insufficient complete historical observations.")

    # Display the number of stocks included in the analysis.
    print("\nDATA PREPARATION")
    print("----------------------------")
    print("Stocks included:", returns.shape[1])

    # Display the number of complete trading-day observations.
    print("Complete trading days:", len(returns))

    # Display the number of excluded stocks, if any.
    print("Excluded stocks:", excluded_columns)

    # Display the historical period used for optimisation.
    print("Start date:", returns.index.min().date())

    # Display the final trading date.
    print("End date:", returns.index.max().date())

    # Return the cleaned daily returns DataFrame.
    return returns


# ============================================================
# 3. CALCULATE EXPECTED RETURNS AND COVARIANCE
# ============================================================

def calculate_market_statistics(returns):

    # Calculate the average daily return of each stock.
    daily_mean_returns = returns.mean()

    # Annualise expected returns using 252 trading days.
    annual_returns = daily_mean_returns * TRADING_DAYS

    # Calculate daily covariance between all stock pairs.
    daily_covariance = returns.cov()

    # Annualise the covariance matrix.
    annual_covariance = daily_covariance * TRADING_DAYS

    # Calculate the correlation between every pair of stocks.
    correlation_matrix = returns.corr()

    # Display the expected annual returns.
    print("\nEXPECTED ANNUAL RETURNS (%)")
    print("----------------------------")
    print((annual_returns * 100).round(2).to_string())

    # Return all three calculated market statistics.
    return annual_returns, annual_covariance, correlation_matrix


# ============================================================
# 4. CALCULATE PORTFOLIO RETURN, RISK AND SHARPE RATIO
# ============================================================

def calculate_portfolio_metrics(
    weights,
    expected_returns,
    covariance
):

    # Convert the expected returns Series into a NumPy array.
    expected_values = expected_returns.to_numpy(dtype=float)

    # Convert the covariance DataFrame into a NumPy matrix.
    covariance_values = covariance.to_numpy(dtype=float)

    # Calculate the weighted annual portfolio return.
    portfolio_return = float(weights @ expected_values)

    # Calculate portfolio variance using the covariance matrix.
    portfolio_variance = float(
        weights @ covariance_values @ weights
    )

    # Calculate annual portfolio volatility.
    portfolio_volatility = np.sqrt(
        max(portfolio_variance, 0.0)
    )

    # Calculate the Sharpe Ratio when portfolio risk is nonzero.
    if portfolio_volatility > 0:

        # Calculate excess return divided by portfolio volatility.
        sharpe_ratio = (
            portfolio_return - RISK_FREE_RATE
        ) / portfolio_volatility

    # Handle the situation where portfolio volatility is zero.
    else:

        # Set Sharpe Ratio to NaN when volatility is zero.
        sharpe_ratio = np.nan

    # Return annual return, volatility and Sharpe Ratio.
    return portfolio_return, portfolio_volatility, sharpe_ratio


# ============================================================
# 5. GENERATE RANDOM PORTFOLIOS
# ============================================================

def simulate_portfolios(
    expected_returns,
    covariance
):

    # Find the total number of stocks.
    number_of_stocks = len(expected_returns)

    # Initialise a reproducible random number generator.
    rng = np.random.default_rng(RANDOM_SEED)

    # Generate random weights that sum to 100% for each portfolio.
    simulated_weights = rng.dirichlet(
        np.ones(number_of_stocks),
        size=NUMBER_OF_SIMULATIONS
    )

    # Convert expected annual returns into a NumPy array.
    expected_values = expected_returns.to_numpy(dtype=float)

    # Convert the covariance matrix into a NumPy array.
    covariance_values = covariance.to_numpy(dtype=float)

    # Calculate the expected annual return of every simulation.
    portfolio_returns = simulated_weights @ expected_values

    # Calculate variance for all simulated portfolios efficiently.
    portfolio_variances = np.einsum(
        "ij,jk,ik->i",
        simulated_weights,
        covariance_values,
        simulated_weights
    )

    # Convert portfolio variance into annualised volatility.
    portfolio_volatilities = np.sqrt(
        np.clip(portfolio_variances, 0, None)
    )

    # Calculate each simulated portfolio's Sharpe Ratio.
    portfolio_sharpes = np.divide(
        portfolio_returns - RISK_FREE_RATE,
        portfolio_volatilities,
        out=np.full_like(portfolio_returns, np.nan),
        where=portfolio_volatilities > 0
    )

    # Create a DataFrame containing simulation results.
    simulation_results = pd.DataFrame({

        # Assign a unique number to every simulated portfolio.
        "Portfolio ID": np.arange(1, NUMBER_OF_SIMULATIONS + 1),

        # Store expected annual returns as percentages.
        "Expected Annual Return (%)": portfolio_returns * 100,

        # Store annualised volatility as percentages.
        "Annualised Volatility (%)": portfolio_volatilities * 100,

        # Store the calculated Sharpe Ratios.
        "Sharpe Ratio": portfolio_sharpes

    })

    # Create a DataFrame containing individual stock weights.
    weight_results = pd.DataFrame(
        simulated_weights,
        columns=expected_returns.index
    )

    # Combine portfolio statistics and individual stock weights.
    simulation_results = pd.concat(
        [simulation_results, weight_results],
        axis=1
    )

    # Return all simulated portfolio results.
    return simulation_results


# ============================================================
# 6. OPTIMISE PORTFOLIO ALLOCATIONS
# ============================================================

def optimise_portfolio(
    expected_returns,
    covariance,
    objective_type
):

    # Determine the number of stocks in the portfolio.
    number_of_stocks = len(expected_returns)

    # Convert expected returns into a NumPy array.
    expected_values = expected_returns.to_numpy(dtype=float)

    # Convert the covariance matrix into a NumPy array.
    covariance_values = covariance.to_numpy(dtype=float)

    # Start with an equal allocation across all stocks.
    initial_weights = np.full(
        number_of_stocks,
        1 / number_of_stocks
    )

    # Set the total investment allocation equal to 100%.
    constraints = [
        {
            "type": "eq",
            "fun": lambda weights: np.sum(weights) - 1
        }
    ]

    # Prevent short selling and restrict each weight to 0%-100%.
    bounds = [(0.0, 1.0)] * number_of_stocks

    # Define the minimum-volatility objective.
    if objective_type == "minimum_volatility":

        # Minimise portfolio variance.
        def objective(weights):

            return float(
                weights @ covariance_values @ weights
            )

    # Define the maximum-Sharpe objective.
    elif objective_type == "maximum_sharpe":

        # Minimise the negative Sharpe Ratio.
        def objective(weights):

            # Calculate the portfolio's annual return.
            portfolio_return = float(
                weights @ expected_values
            )

            # Calculate portfolio variance.
            portfolio_variance = float(
                weights @ covariance_values @ weights
            )

            # Calculate portfolio volatility.
            portfolio_volatility = np.sqrt(
                max(portfolio_variance, 0.0)
            )

            # Avoid division by zero.
            if portfolio_volatility <= 0:
                return 1e6

            # Calculate the portfolio Sharpe Ratio.
            sharpe = (
                portfolio_return - RISK_FREE_RATE
            ) / portfolio_volatility

            # Return negative Sharpe because minimize is used.
            return -sharpe

    # Reject unsupported optimisation objectives.
    else:

        raise ValueError("Unknown portfolio optimisation objective.")

    # Run Sequential Least Squares Programming optimisation.
    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 2000,
            "ftol": 1e-10
        }
    )

    # Check whether the numerical optimisation succeeded.
    if not result.success:

        # Stop if a reliable optimal solution was not obtained.
        raise RuntimeError(
            f"Portfolio optimisation failed: {result.message}"
        )

    # Retrieve the optimal stock weights.
    optimal_weights = np.clip(result.x, 0, 1)

    # Normalise weights to remove insignificant floating-point error.
    optimal_weights = optimal_weights / optimal_weights.sum()

    # Return the optimal allocation.
    return optimal_weights


# ============================================================
# 7. GENERATE THE EFFICIENT FRONTIER
# ============================================================

def generate_efficient_frontier(
    expected_returns,
    covariance,
    number_of_points=50
):

    # Convert expected returns into a NumPy array.
    expected_values = expected_returns.to_numpy(dtype=float)

    # Find the minimum expected return among the stocks.
    minimum_return = float(expected_values.min())

    # Find the maximum expected return among the stocks.
    maximum_return = float(expected_values.max())

    # Identify the stock with the minimum expected return.
    minimum_index = int(np.argmin(expected_values))

    # Identify the stock with the maximum expected return.
    maximum_index = int(np.argmax(expected_values))

    # Generate target returns between the minimum and maximum.
    target_returns = np.linspace(
        minimum_return,
        maximum_return,
        number_of_points
    )

    # Convert the covariance DataFrame into a NumPy array.
    covariance_values = covariance.to_numpy(dtype=float)

    # Create a list to store efficient frontier results.
    frontier_results = []

    # Process each target expected portfolio return.
    for target_return in target_returns:

        # Start with zero allocations to every stock.
        initial_weights = np.zeros(len(expected_values))

        # Calculate the difference between maximum and minimum returns.
        return_range = maximum_return - minimum_return

        # Create a feasible starting portfolio for this target return.
        if return_range > 1e-12:

            # Calculate the weight of the minimum-return stock.
            min_weight = (
                maximum_return - target_return
            ) / return_range

            # Allocate the remaining weight to the maximum-return stock.
            max_weight = 1 - min_weight

            # Assign the minimum-return stock's allocation.
            initial_weights[minimum_index] = min_weight

            # Assign the maximum-return stock's allocation.
            initial_weights[maximum_index] = max_weight

        # Handle the case where all expected returns are equal.
        else:

            # Begin with an equal allocation.
            initial_weights = np.full(
                len(expected_values),
                1 / len(expected_values)
            )

        # Require the sum of all weights to equal 100%.
        constraints = [
            {
                "type": "eq",
                "fun": lambda weights: np.sum(weights) - 1
            },

            # Require the portfolio to meet its target return.
            {
                "type": "eq",
                "fun": lambda weights, target=target_return:
                    float(weights @ expected_values - target)
            }
        ]

        # Define the minimum-variance objective.
        def objective(weights):

            # Return the portfolio variance for the given weights.
            return float(
                weights @ covariance_values @ weights
            )

        # Find the minimum-variance allocation for this target return.
        result = minimize(
            objective,
            initial_weights,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * len(expected_values),
            constraints=constraints,
            options={
                "maxiter": 2000,
                "ftol": 1e-10
            }
        )

        # Retain only solutions that satisfy the optimiser's success check.
        if result.success:

            # Calculate the achieved portfolio return and volatility.
            achieved_return, volatility, sharpe = (
                calculate_portfolio_metrics(
                    result.x,
                    expected_returns,
                    covariance
                )
            )

            # Store the successful efficient frontier point.
            frontier_results.append({

                # Store the target annual expected return.
                "Target Return (%)": target_return * 100,

                # Store the achieved annual return.
                "Expected Annual Return (%)": achieved_return * 100,

                # Store the portfolio's annualised volatility.
                "Annualised Volatility (%)": volatility * 100,

                # Store the portfolio Sharpe Ratio.
                "Sharpe Ratio": sharpe

            })

    # Convert the efficient frontier results into a DataFrame.
    frontier_df = pd.DataFrame(frontier_results)

    # Ensure at least one frontier solution was successfully generated.
    if frontier_df.empty:

        # Stop rather than save an empty frontier.
        raise RuntimeError("Unable to generate efficient frontier.")

    # Return the efficient frontier DataFrame.
    return frontier_df


# ============================================================
# 8. SAVE PORTFOLIO RESULTS
# ============================================================

def save_portfolio_outputs(
    correlation_matrix,
    expected_returns,
    simulations,
    optimal_weights,
    portfolio_summary,
    efficient_frontier
):

    # Create the portfolio output directory if it does not exist.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save the stock correlation matrix.
    correlation_matrix.to_csv(
        OUTPUT_DIR / "correlation_matrix.csv"
    )

    # Save expected annual stock returns.
    expected_returns.rename(
        "Expected Annual Return (Decimal)"
    ).to_csv(
        OUTPUT_DIR / "expected_returns.csv"
    )

    # Save all simulated portfolio results and weights.
    simulations.to_csv(
        OUTPUT_DIR / "portfolio_simulations.csv",
        index=False
    )

    # Save allocations for the two optimised portfolios.
    optimal_weights.to_csv(
        OUTPUT_DIR / "optimal_portfolio_weights.csv",
        index=False
    )

    # Save minimum-risk and maximum-Sharpe portfolio summaries.
    portfolio_summary.to_csv(
        OUTPUT_DIR / "portfolio_summary.csv",
        index=False
    )

    # Save the efficient frontier points.
    efficient_frontier.to_csv(
        OUTPUT_DIR / "efficient_frontier.csv",
        index=False
    )

    # Display the directory containing all saved results.
    print("\nPORTFOLIO OUTPUTS SAVED TO:")
    print(OUTPUT_DIR)


# ============================================================
# 9. PLOT THE EFFICIENT FRONTIER
# ============================================================
def plot_efficient_frontier(
    simulations,
    efficient_frontier,
    portfolio_summary
):
    # Create a figure and axis for portfolio visualisation.
    fig, ax = plt.subplots(figsize=(11, 7))

    # Plot the 10,000 simulated portfolios.
    scatter = ax.scatter(
        simulations["Annualised Volatility (%)"],
        simulations["Expected Annual Return (%)"],
        c=simulations["Sharpe Ratio"],
        s=12,
        alpha=0.5
    )

    # Prepare the efficient frontier for plotting.
    frontier_plot = efficient_frontier[
        [
            "Annualised Volatility (%)",
            "Expected Annual Return (%)"
        ]
    ].replace([np.inf, -np.inf], np.nan).dropna()

    # Sort portfolios by ascending risk.
    # For identical risk values, retain the portfolio with the highest return.
    frontier_plot = frontier_plot.sort_values(
        ["Annualised Volatility (%)", "Expected Annual Return (%)"],
        ascending=[True, False]
    ).drop_duplicates(
        subset=["Annualised Volatility (%)"],
        keep="first"
    ).reset_index(drop=True)

    # Remove dominated portfolios.
    previous_best_return = (
        frontier_plot["Expected Annual Return (%)"]
        .cummax()
        .shift(1)
    )

    frontier_plot = frontier_plot.loc[
        previous_best_return.isna()
        | (
            frontier_plot["Expected Annual Return (%)"]
            > previous_best_return
        )
    ]

    # Plot the properly ordered efficient frontier.
    ax.plot(
        frontier_plot["Annualised Volatility (%)"],
        frontier_plot["Expected Annual Return (%)"],
        marker="o",
        markersize=3,
        linewidth=2,
        label="Efficient Frontier"
    )

    # Select the minimum-volatility portfolio.
    minimum_row = portfolio_summary[
        portfolio_summary["Portfolio"] == "Minimum Volatility"
    ].iloc[0]

    # Select the maximum-Sharpe portfolio.
    maximum_row = portfolio_summary[
        portfolio_summary["Portfolio"] == "Maximum Sharpe"
    ].iloc[0]

    # Mark the minimum-volatility portfolio.
    ax.scatter(
        minimum_row["Annualised Volatility (%)"],
        minimum_row["Expected Annual Return (%)"],
        marker="*",
        s=250,
        label="Minimum Volatility"
    )

    # Mark the maximum-Sharpe portfolio.
    ax.scatter(
        maximum_row["Annualised Volatility (%)"],
        maximum_row["Expected Annual Return (%)"],
        marker="*",
        s=250,
        label="Maximum Sharpe"
    )

    # Label the horizontal axis.
    ax.set_xlabel("Annualised Volatility (%)")

    # Label the vertical axis.
    ax.set_ylabel("Expected Annual Return (%)")

    # Add a descriptive chart title.
    ax.set_title("Modern Portfolio Theory - Efficient Frontier")

    # Add a colour bar representing the Sharpe Ratio.
    colorbar = fig.colorbar(scatter, ax=ax)

    # Label the colour bar.
    colorbar.set_label("Sharpe Ratio")

    # Display the chart legend.
    ax.legend()

    # Add a grid to improve chart readability.
    ax.grid(alpha=0.3)

    # Adjust spacing around the chart.
    plt.tight_layout()

    # Save the chart to the portfolio output directory.
    fig.savefig(
        OUTPUT_DIR / "efficient_frontier.png",
        dpi=160
    )

    # Close the figure to release memory.
    plt.close(fig)

    # Display the saved chart location.
    print("\nEfficient frontier chart saved successfully.")
# ============================================================
# 10. RUN THE COMPLETE PORTFOLIO OPTIMISATION
# ============================================================

def run_portfolio_optimization():

    # Display the beginning of the optimisation process.
    print("\n======================================")
    print(" STOCK PORTFOLIO OPTIMISATION")
    print("======================================")

    # Load and validate the historical stock returns.
    returns = load_daily_returns()

    # Calculate annual expected returns, covariance and correlation.
    expected_returns, covariance, correlation = (
        calculate_market_statistics(returns)
    )

    # Simulate 10,000 possible investment allocations.
    simulations = simulate_portfolios(
        expected_returns,
        covariance
    )

    # Calculate the minimum-volatility portfolio weights.
    minimum_weights = optimise_portfolio(
        expected_returns,
        covariance,
        "minimum_volatility"
    )

    # Calculate the maximum-Sharpe portfolio weights.
    maximum_weights = optimise_portfolio(
        expected_returns,
        covariance,
        "maximum_sharpe"
    )

    # Calculate the financial metrics of the minimum-risk portfolio.
    min_return, min_volatility, min_sharpe = (
        calculate_portfolio_metrics(
            minimum_weights,
            expected_returns,
            covariance
        )
    )

    # Calculate the financial metrics of the maximum-Sharpe portfolio.
    max_return, max_volatility, max_sharpe = (
        calculate_portfolio_metrics(
            maximum_weights,
            expected_returns,
            covariance
        )
    )

    # Generate the efficient frontier.
    efficient_frontier = generate_efficient_frontier(
        expected_returns,
        covariance
    )

    # Create a summary of both optimised portfolios.
    portfolio_summary = pd.DataFrame([

        # Store the minimum-volatility portfolio results.
        {
            "Portfolio": "Minimum Volatility",
            "Expected Annual Return (%)": min_return * 100,
            "Annualised Volatility (%)": min_volatility * 100,
            "Sharpe Ratio": min_sharpe
        },

        # Store the maximum-Sharpe portfolio results.
        {
            "Portfolio": "Maximum Sharpe",
            "Expected Annual Return (%)": max_return * 100,
            "Annualised Volatility (%)": max_volatility * 100,
            "Sharpe Ratio": max_sharpe
        }

    ])

    # Create a table containing each stock's optimal allocation.
    optimal_weights = pd.DataFrame({

        # Store the names of the 20 stocks.
        "Ticker": expected_returns.index,

        # Store each stock's historical expected annual return.
        "Expected Annual Return (%)": expected_returns.values * 100,

        # Store each stock's minimum-volatility allocation.
        "Minimum Volatility Weight (%)": minimum_weights * 100,

        # Store each stock's maximum-Sharpe allocation.
        "Maximum Sharpe Weight (%)": maximum_weights * 100

    })

    # Save all generated portfolio datasets.
    save_portfolio_outputs(
        correlation,
        expected_returns,
        simulations,
        optimal_weights,
        portfolio_summary,
        efficient_frontier
    )

    # Generate and save the efficient frontier visualisation.
    plot_efficient_frontier(
        simulations,
        efficient_frontier,
        portfolio_summary
    )

    # Display the portfolio performance comparison.
    print("\nOPTIMISED PORTFOLIO COMPARISON")
    print("----------------------------")
    print(portfolio_summary.round(4).to_string(index=False))

    # Display the maximum-Sharpe portfolio's allocations.
    print("\nMAXIMUM SHARPE PORTFOLIO ALLOCATIONS")
    print("-----------------------------------")

    # Sort stocks from largest to smallest allocation.
    print(
        optimal_weights[
            ["Ticker", "Maximum Sharpe Weight (%)"]
        ]
        .sort_values(
            "Maximum Sharpe Weight (%)",
            ascending=False
        )
        .round(2)
        .to_string(index=False)
    )

    # Display the minimum-volatility portfolio's allocations.
    print("\nMINIMUM VOLATILITY PORTFOLIO ALLOCATIONS")
    print("----------------------------------------")

    # Sort stocks from largest to smallest allocation.
    print(
        optimal_weights[
            ["Ticker", "Minimum Volatility Weight (%)"]
        ]
        .sort_values(
            "Minimum Volatility Weight (%)",
            ascending=False
        )
        .round(2)
        .to_string(index=False)
    )

    # Display a final completion message.
    print("\nPORTFOLIO OPTIMISATION COMPLETED SUCCESSFULLY.")


# ============================================================
# 11. ALLOW DIRECT TERMINAL EXECUTION
# ============================================================

# Execute the optimisation only when this file is run directly.
if __name__ == "__main__":

    # Start the complete portfolio optimisation pipeline.
    run_portfolio_optimization()
