import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import minimize

matplotlib.use('TkAgg')

# Global parameters
window = 180

# Read in data
data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Market_concentration_risk_paper\top_100_data_\combined_prices.csv")
# Set index of the data
data.index = data.iloc[:, 0]
df = data.iloc[:, 1:]
df_clean = df.dropna(axis=1)
# Compute log returns of the market
df_returns = np.log(df_clean) - np.log(df_clean).shift(1)

# Rolling window portfolio optimization function
def rolling_portfolio_optimization(df_returns, window):
    num_assets = df_returns.shape[1]
    optimisation_weights = []

    for i in range(window, len(df_returns)): # len(df_returns)
        print("Iteration ", i)
        # Get the slice of data for the current window
        returns_window = df_returns.iloc[i-window:i, :]

        # Compute mean returns (handling NaN)
        mean_returns = returns_window.mean()
        mean_returns[np.isnan(mean_returns)] = 0.0  # Replace NaN with 0

        # Compute covariance matrix (handling NaN)
        cov_matrix = returns_window.cov()
        cov_matrix[np.isnan(cov_matrix)] = 0.0  # Replace NaN with 0

        # Define optimization function (negative of portfolio return)
        def objective_function(weights):
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_variance = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return portfolio_return/portfolio_variance

        # Define constraints (sum of weights equals 1)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                       {'type': 'ineq', 'fun': lambda x: x - 0.0005},  # Minimum weight constraint
                       {'type': 'ineq', 'fun': lambda x: 0.05 - x})   # Maximum weight constraint

        # Define initial weights (equal weights)
        initial_weights = np.ones(num_assets) / num_assets
        # Perform optimization
        result = minimize(objective_function, initial_weights, method='SLSQP', constraints=constraints)
        # Store optimized weights
        optimisation_weights.append(result.x)

    return np.array(optimisation_weights)

# Example usage:
# Perform rolling window portfolio optimization
rolling_opt_weights = rolling_portfolio_optimization(df_returns, window)
# Create dataframe
rolling_opt_weights_df = pd.DataFrame(rolling_opt_weights)
rolling_opt_weights_df.index = df_clean.index[window:]
# Get columns
rolling_opt_weights_df.columns = df_clean.columns
rolling_opt_weights_df.to_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Market_concentration_risk_paper\top_100_data_\Optimal_weights_df.csv")
