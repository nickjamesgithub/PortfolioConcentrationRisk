import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage

matplotlib.use('TkAgg')

make_plots = True

# Import data
df = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Market_concentration_risk_paper\Market_Capitalisation_Data.csv")
df.index = df["Name"]
df = df.iloc[:,1:]

# Replace 'data unavailable' with NaN
df.replace('Data Unavailable', np.nan, inplace=True)
# Convert columns to numeric, forcing errors to NaN
df = df.apply(pd.to_numeric, errors='coerce')
df_clean = df.dropna(axis=0)

# Compute total market capitalization for each date
total_market_cap = df_clean.sum(axis=0, skipna=True)

# Define different values of top_n to consider
top_n_values = [3, 5, 10, 15]
colors = ['blue', 'green', 'red', 'purple']

# Create a figure and axis
plt.figure(figsize=(12, 6))

# Loop over each top_n value
for top_n, color in zip(top_n_values, colors):
    # Compute the market capitalization of the top_n companies
    top_market_cap = df_clean.apply(lambda x: x.nlargest(top_n).sum(), axis=0)
    # Compute the percentage of market capitalization of the top_n companies
    top_market_cap_percentage = (top_market_cap / total_market_cap) * 100
    # Apply adjustment factor
    adjusted_top_market_cap_percentage = top_market_cap_percentage
    # Plot the time-varying percentage
    plt.plot(df.columns, adjusted_top_market_cap_percentage, label=f'Top {top_n}', color=color)

if make_plots:
    # Customize the plot
    plt.xlabel('Date')
    plt.ylabel('Percentage of market')
    plt.title('Time-Varying Market Value Concentration Percentage')
    plt.legend()
    # Limit number of tickers on x-axis
    n_dates = 10  # Number of dates to display
    plt.xticks(np.arange(1, len(df.columns), len(df.columns)//n_dates), rotation=90)
    # Show the plot
    plt.tight_layout()
    plt.savefig("Market_value_concentration")
    plt.show()

# Raw data
data_slice = df.iloc[:,1:]
# Get dates
dates = data_slice.columns
# Get company names
names = data_slice.iloc[:,0]
# Transpose the matrix
data_slice_transpose = data_slice.transpose()

# Initialise distance matrix
dist_matrix = np.zeros(((len(data_slice_transpose), len(data_slice_transpose))))
for i in range(len(data_slice_transpose)):
    for j in range(len(data_slice_transpose)):
        dist_i = np.nan_to_num(data_slice_transpose.iloc[i,:].values)
        dist_i_weights = dist_i/np.sum(dist_i)
        dist_j = np.nan_to_num(data_slice_transpose.iloc[j,:].values)
        dist_j_weights = dist_j / np.sum(dist_j)
        dist_w = wasserstein_distance(dist_i, dist_j, dist_i_weights, dist_j_weights)
        # Fill respective element on distance matrix
        dist_matrix[i,j] = dist_w
    print("Iteration D(S,T)", i)

if make_plots:
    # Plot the heatmap
    plt.matshow(dist_matrix)
    # Select only every 20th date for labeling
    step = 20
    selected_dates = dates[::step]
    selected_ticks = np.arange(0, len(dates), step)
    # Add selected dates as labels on x and y axes
    plt.xticks(ticks=selected_ticks, labels=selected_dates, rotation=90)
    plt.yticks(ticks=selected_ticks, labels=selected_dates)
    # Add colorbar for reference
    plt.colorbar()
    # Add titles and labels
    plt.title('Wasserstein Distance Heatmap')
    plt.xlabel('Dates')
    plt.ylabel('Dates')
    # Show the plot
    plt.savefig("D_ST_Market_Capitalisation")
    plt.show()

# Generate the linkage matrix
df_dist = pd.DataFrame(dist_matrix, index = dates, columns = dates)
# Generate the linkage matrix
linkage_matrix = linkage(dist_matrix, method='ward')
# Create the clustermap
sns.clustermap(df_dist, row_linkage=linkage_matrix, col_linkage=linkage_matrix, cmap='viridis')
# Show the plot
plt.savefig("D_ST_Market_Capitalisation_Clustering")
plt.show()
