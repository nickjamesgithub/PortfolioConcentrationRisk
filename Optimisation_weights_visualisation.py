import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import wasserstein_distance
from matplotlib.ticker import MaxNLocator
from scipy.signal import savgol_filter
from Utilities import dendrogram_plot_test

matplotlib.use('TkAgg')

make_heatmap = False

# Read in optimal weights
opt_weights = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Market_concentration_risk_paper\top_100_data_\Optimal_weights_df.csv")
opt_weights = opt_weights.iloc[:, 1:]

# Read in original data
data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Market_concentration_risk_paper\top_100_data_\combined_prices.csv")
# Set index of the data
data.index = data.iloc[:, 0]
df = data.iloc[:, 1:]

if make_heatmap:
    dist_matrix = np.zeros(((len(opt_weights), len(opt_weights))))
    for i in range(len(opt_weights)):
        for j in range(len(opt_weights)):
            dist_matrix[i, j] = wasserstein_distance(opt_weights.iloc[i, :], opt_weights.iloc[j, :])
        print("Iteration ", i)

    # Visualise optimisation surface
    plt.matshow(dist_matrix)
    plt.colorbar()
    plt.savefig("Optimisation_weights_topology")
    plt.show()

# Increase figure size and adjust aspect ratio
plt.figure(figsize=(12, 8))  # Adjust the figure size as needed
plt.imshow(opt_weights, cmap='hot', interpolation='nearest', aspect='auto')  # Use 'auto' aspect ratio
plt.colorbar()
plt.title('Optimal Weights Heatmap')
plt.xlabel('Assets')
plt.ylabel('Dates')
plt.savefig("Optimal_weight_evolution")
plt.show()

# Add column labels from data at the top of the heatmap
plt.xticks(ticks=np.arange(len(df.columns)), labels=df.columns, rotation=90, fontsize=6)
# Limit the number of x-tick values to 5
ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
# Calculate the time series of differenced vector norms
diff_vector_norms = np.sum(np.abs(opt_weights.diff().dropna()), axis=1)
# Include Savitzky Golay filter
smoothed_vector_norm = savgol_filter(diff_vector_norms, 201, 1)
# Plot the time series of differenced vector norms
plt.figure(figsize=(12, 6))
plt.plot(data.index[181:], diff_vector_norms, label='Differenced Vector Norms')
plt.plot(data.index[181:], smoothed_vector_norm, label='Denoised Signal')
plt.title('First differenced vector norm and smoothed function')
plt.xlabel('Time')
plt.ylabel('Sum of Absolute Differences')
plt.legend()
# Limit the number of x-tick values to 5 for the time series plot
ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
plt.savefig("Vector_norm_diff")
plt.show()

# Loop over the columns (these are companies)
association_matrix = np.zeros(((len(opt_weights.iloc[0,:]), len(opt_weights.iloc[0,:]))))
for i in range(len(opt_weights.iloc[0,:])):
    for j in range(len(opt_weights.iloc[0, :])):
        weight_vector_i = opt_weights.iloc[:,i]
        weight_vector_j = opt_weights.iloc[:, j]
        inner_product = np.dot(weight_vector_i, weight_vector_j)/(np.linalg.norm(weight_vector_i) * np.linalg.norm(weight_vector_j))
        association_matrix[i,j] = inner_product
    print("Iteration ", i)

# Plot association matrix
plt.matshow(association_matrix)
plt.colorbar()
plt.show()

# High association examples
plt.plot(opt_weights["GS"], alpha = 0.25)
plt.plot(opt_weights["MS"], alpha = 0.25)
plt.title("High association weight vector")
plt.show()

# Plot dendrogram
dendrogram_plot_test(association_matrix, "weight_allocation_",  "inner_product", np.array(opt_weights.columns))