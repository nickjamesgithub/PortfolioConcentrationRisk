import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

# Turn feature engineering label on
feature_engineering = True

# Import data
df = pd.read_parquet(r'C:\Users\60848\OneDrive - Bain\Desktop\Telstra\Bain_data.parquet')
# Get the most recent data for each unique customer (churned/non-churned)
df['MONTH_DT'] = pd.to_datetime(df['MONTH_DT'])

# Define the columns to encode
columns_to_encode = ['MMC', "Fxd_Voice_Cn", "Fxd_Brdbd_Cn", "Tenure_Yrs"]

# Get unique months
unique_months = np.sort(pd.to_datetime(df['MONTH_DT'].unique()))

# Initialize a dictionary to hold the sums of the absolute values above the diagonal for each column over time
upper_tri_mean = {col: [] for col in columns_to_encode}

for col in columns_to_encode:
    for month in range(len(unique_months) - 1):
        # Filter the dataframe for the current month
        df_month = df[df['MONTH_DT'] == unique_months[month]].fillna(0)

        # Extract the vector for the current column
        vector = df_month[col].values.reshape(-1, 1)
        # Compute the L1 norm distance matrix
        l1_distance_matrix = np.abs(vector - vector.T)
        affinity_matrix = 1 - l1_distance_matrix/np.max(l1_distance_matrix)

        # Get the upper triangle indices, excluding the diagonal
        triu_indices = np.triu_indices_from(affinity_matrix, k=1)

        # Extract the elements above the diagonal
        upper_tri_elements = affinity_matrix[triu_indices]

        # Compute the sum of the absolute values of the elements above the diagonal
        upper_tri_sum = np.mean(upper_tri_elements)

        # Append the sum to the list for the current column
        upper_tri_mean[col].append(upper_tri_sum)

        print(f"Iteration {unique_months[month]} for column {col}: Sum of upper triangle = {upper_tri_sum}")

# Plot the sums of the absolute values above the diagonal over time for each column
plt.figure(figsize=(12, 8))
for col in columns_to_encode:
    plt.plot(unique_months[:-1], upper_tri_mean[col], label=col)

plt.title("Sum of Absolute Values Above Diagonal Over Time for Each Feature")
plt.xlabel("Month")
plt.ylabel("Mean of Absolute Values Above Diagonal")
plt.legend(loc="upper right")
plt.savefig("Upper_Triangle_Sum_Evolution.png")
plt.show()