import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import eigh

matplotlib.use('TkAgg')

# Global parameters
window = 90

# Read in data
data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Market_concentration_risk_paper\top_100_data_\combined_prices.csv")
# Set index of the data
data.index = data.iloc[:, 0]
df = data.iloc[:, 1:]
# Compute log returns of the market
df_returns = np.log(df) - np.log(df).shift(1)

# Initialize a list to store the explanatory variance for the first 10 eigenvalues
leading_eigenvalues_list = []

for i in range(window, len(df_returns)):
    # Iteration and date
    print("Iteration " , i, " Date ", df_returns.index[i])
    # Identify rolling window
    df_slice_i = df_returns.iloc[(i - window):i, :]
    # Clean the log returns
    df_clean_i = df_slice_i.dropna(axis=1, how='all')
    # Compute correlation matrix
    correlation_matrix = np.nan_to_num(df_clean_i.corr())

    # Perform eigendecomposition
    m_vals, m_vecs = eigh(correlation_matrix)

    # Calculate the explanatory variance for the first 10 eigenvalues
    for j in range(1, 11):  # 1 to 10
        m_vals_j = m_vals[-j] / len(correlation_matrix)
        leading_eigenvalues_list.append(m_vals_j)

# To organize the leading eigenvalues list into a dataframe or another suitable structure
leading_eigenvalues_df = pd.DataFrame(np.array(leading_eigenvalues_list).reshape(-1, 10), columns=[f'{k + 1}' for k in range(10)])
print(leading_eigenvalues_df)

# Get dates for the x-axis
date_range = df_returns.index[window:len(df_returns)]

# Create meshgrid for dates (x-axis) and eigenvalues (y-axis)
X, Y = np.meshgrid(np.arange(len(date_range)), np.arange(1, 11))

# Create the figure and axis
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')
# Set vmin and vmax based on the data
vmin = leading_eigenvalues_df.values.min()
vmax = leading_eigenvalues_df.values.max()

# Create the surface plot
surf = ax.plot_surface(X, Y, leading_eigenvalues_df.T, cmap='plasma', vmin=vmin, vmax=vmax)

# Add color bar
fig.colorbar(surf)

# Set x-axis labels
ax.set_xticks(np.arange(len(date_range)))
ax.set_xticklabels(date_range, rotation=30, ha='left')

# Set y-axis labels to show only a few ticks
ax.set_yticks(np.arange(1, 11))
ax.set_yticklabels(leading_eigenvalues_df.columns)

# Limit number of date ticks for readability
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
ax.yaxis.set_major_locator(plt.MaxNLocator(5))

# Set the viewing angle
ax.view_init(elev=30, azim=45)  # Change these values to set the desired angle

# Labels and title
ax.set_xlabel('Date')
ax.set_ylabel('Eigenvalues')
ax.set_zlabel('Explanatory Variance')
ax.set_title('Eigenvalue Explanatory Variance vs Time')
# Save the figure
plt.savefig("Eigenvalue_explanatory_variance.png")
# Show the plot
plt.show()

