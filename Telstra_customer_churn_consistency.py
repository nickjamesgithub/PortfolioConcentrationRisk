import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cityblock  # For L1 distance

matplotlib.use('TkAgg')

# Import data
df = pd.read_parquet(r'C:\Users\60848\OneDrive - Bain\Desktop\Telstra\Bain_data.parquet')

# Define the date threshold
df['MONTH_DT'] = pd.to_datetime(df['MONTH_DT'], errors='coerce')
# Check for any non-convertible dates and handle them (optional)
non_convertible_dates = df['MONTH_DT'].isna().sum()
if non_convertible_dates > 0:
    print(f"There are {non_convertible_dates} non-convertible dates in the MONTH_DT column.")
# Define the date threshold
date_threshold = pd.to_datetime('2023-12-01')
# Filter the dataframe
filtered_df = df[(df['FBB_Churn_fla'] == 1) & (df['MONTH_DT'] < date_threshold)]
# Count the number of customers
num_customers = filtered_df.shape[0]

# Get the most recent data for each unique customer (churned/non-churned)
df['MONTH_DT'] = pd.to_datetime(df['MONTH_DT'])

# Churned customers
churned_id = df.loc[df["FBB_Churn_fla"] == 1]["Anon_srvc_id"].values
# Initialize an empty DataFrame to store the similarities
all_similarity_df = pd.DataFrame()
# Loop over all churned customers and then compute
churned_customer_observations = []
for i in range(len(churned_id)):
    print("Iteration ", i)
    # Slice data based on churn
    slice = df.loc[df["Anon_srvc_id"] == churned_id[i]]
    # Churned customer observation
    churned_customer_observations.append(len(slice))
    # Assuming 'slice' is your DataFrame
    unique_counts = slice.nunique()
    total_rows = len(slice)

    # Calculate percentage similarity
    percentage_similarity = (slice.apply(lambda x: x.value_counts(normalize=True).max()) * 100).round(2)

    # Convert percentage_similarity to DataFrame
    similarity_df = pd.DataFrame({
        'Column': percentage_similarity.index,
        'Percentage Similarity': percentage_similarity.values
    })

    # Append the similarity_df to all_similarity_df
    all_similarity_df = pd.concat([all_similarity_df, similarity_df], ignore_index=True)

# Compute the mean for the "Percentage of Similarity" for each column
mean_similarity_df = all_similarity_df.groupby('Column').mean().reset_index()
# Print the mean similarities DataFrame
print(mean_similarity_df)

# Distribution of months until customer churn
plt.hist(churned_customer_observations)
plt.xlabel("Months until churn")
plt.ylabel("Frequency")
plt.savefig("Months_before_churn_distribution")
plt.show()