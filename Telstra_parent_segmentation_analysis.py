import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from pandas.api.types import CategoricalDtype
matplotlib.use('TkAgg')

# Import data
data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Telstra\Data_with_cluster_projections_2d.csv",
                   low_memory=False)

# Example mapping tables for each question; replace these with your actual mappings
mappings = {
    "Q1_encode": {1: "18-24", 2: "25-34", 3: "35-44", 4: "45-54", 5: "55-64", 6: "65-74", 7: "75 and above"},
    "Q4_encode": {1: 'Less than $15,000', 2: '$15,000 to $29,999', 3: '$30,000 to $44,999', 4: '$45,000 to $59,999',
                  5: '$60,000 to $79,999', 6: '$80,000 to $99,999', 7: '$100,000 to $119,999',
                  8: '$120,000 to $149,999', 9: '$150,000 to $199,999', 10: '$200,000 or more'},
    "Q8_encode": {1: "PS by need", 2: "PS by choice", 3: "Premium", 4: "Iconic"},
    "Q14_encode": {0: "No rank", 100: "Ranked_first", 40: "Ranked_second", 20: "Ranked_third", 10: "Ranked_fourth",
                   5: "Ranked_fifth"},
    "Q16_encode": {1: 'I don’t know', 2: '4G wireless broadband (fixed wireless)',
                   3: '5G wireless broadband (fixed wireless)',
                   4: '12 mbps', 5: '25 mbps', 6: '50 mbps', 7: '100 mbps', 8: '250 mbps',
                   9: '500 mbps', 10: '750 mbps', 11: '1,000 mbps'},
    "Q17_encode": {1: 'I don’t know', 2: 'Less than $50', 3: '$50 - $54.99', 4: '$55 - $59.99', 5: '$60 - $64.99',
                   6: '$65 - $69.99', 7: '$70 - $74.99', 8: '$75 - $79.99', 9: '$80 - $84.99',
                   10: '$85 - $89.99', 11: '$90 - $94.99', 12: '$95 - $99.99', 13: '$100 - $104.99',
                   14: '$105 - $109.99', 15: '$110 - $114.99', 16: '$115 - $119.99', 17: '$120 - $124.99',
                   18: '$125 - $129.99', 19: '$130 - $134.99', 20: '$135 - $139.99', 21: '$140 - $144.99',
                   22: '$145 - $149.99', 23: '$150 or more'},
    "Q23_encode": {1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
    "Q30_encode": {0: 'Less than 6 months ', 1: '6-12 months ', 2: '12-24 months', 3: '2-4 years',
                   4: 'More than 4 years',
                   5: "5-7 years", 6: "8-10 years", 7: "Over 10 years"},
    "Q31_encode": {0: "Never", 1: "Over 10 years", 2: "8-10 years ago", 3: "5-7 years ago", 4: ">4 Years ago",
                   5: "2-4 years ago", 6: "12-24 months ago", 7: "6-12 months", 8: "last 6 months"}
}

# Define which prefixes to apply mode
mode_prefixes = ["Q1_encode", "Q4_encode", "Q14_encode", "Q16_encode", "Q17_encode", "Q23_encode", "Q30_encode",
                 "Q31_encode", "Q35_encode", "Q41_encode", "Q42_encode"]

# Prepare an empty list to store the summary data
summary_table_list = []
column_keys = []

# Calculate value counts for each column in the specified prefixes by cluster
for prefix in mode_prefixes:
    cols = [col for col in data.columns if col.startswith(prefix)]
    for col in cols:
        # Group by cluster and calculate normalized value counts (percentages)
        value_counts = data.groupby('Cluster_Label')[col].value_counts(normalize=True).unstack(fill_value=0)

        # Apply mapping to each column based on the original column's prefix
        if prefix in mappings:
            value_counts.index = value_counts.index.map(mappings[prefix])

        # Reset index to avoid conflicts during concatenation
        value_counts.reset_index(inplace=True, drop=True)

        # Append the value counts for this column to the summary table list
        summary_table_list.append(value_counts)
        column_keys.append(col)

# Combine all the summary tables into one DataFrame
summary_table = pd.concat(summary_table_list, axis=1, keys=column_keys)

# Display the summary table with percentages
pd.set_option('display.max_columns', None)  # Display all columns
print(summary_table)

# Step 1: Filter columns that start with "Q14_encode"
q14_columns = [col for col in data.columns if col.startswith("Q14_encode")]
# Step 2: Group by 'cluster_Label' and calculate the mean for the filtered columns
kpc_propensities = data.groupby('Cluster_Label')[q14_columns].mean()
kpc_propensities.to_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Telstra\KPC_propensities_by_cluster.csv")
# Display the resulting DataFrame with average values
print(kpc_propensities)

# Define the two columns for the joint distribution
col1 = 'Q17_encode'
col2 = 'Q42_encode'

# Ensure that all values in Q17_encode have corresponding mappings
def map_value(x):
    try:
        return next(k for k, v in mappings[col1].items() if v == x)
    except StopIteration:
        return None  # Handle unmapped values by returning None or a default value

# Apply the mapping to the 'Q17_encode' column in the original DataFrame
data[col1] = data[col1].map(mappings[col1])
data['Q17_mapped'] = data[col1].map(map_value)

# Remove rows with unmapped values if necessary
data = data.dropna(subset=['Q17_mapped'])

# Automatically create 5 bins for Q17_encode and Q42_encode using qcut, handling duplicates
data['Q17_binned'] = pd.qcut(data['Q17_mapped'], q=5, duplicates='drop')
data['Q42_binned'] = pd.qcut(data[col2], q=5, duplicates='drop')

# Get unique cluster labels and sort them
cluster_labels = sorted(data['Cluster_Label'].unique())

# # Loop over each cluster in order
# for cluster in cluster_labels:
#     # Filter data for the current cluster
#     cluster_data = data[data['Cluster_Label'] == cluster]
#
#     # Compute the joint distribution (cross-tabulation) of the binned variables
#     joint_distribution = pd.crosstab(cluster_data['Q17_binned'], cluster_data['Q42_binned'], normalize='all') * 100
#
#     # Display the joint distribution table as a DataFrame with axis labels
#     print(f'Joint Distribution of {col1} and {col2} for Cluster {cluster}')
#     print(f'Y-Axis (Rows): {col1}')
#     print(f'X-Axis (Columns): {col2}')
#     print(joint_distribution)
#     print("\n")  # Add space between clusters for readability
#
#     # Optionally, save the table to a CSV file
#     joint_distribution.to_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Telstra\Joint_distribution_table_cluster+"+str(cluster)+".csv")
#

import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import CategoricalDtype

# Define the columns for speed tiers and prices
speed_col = 'Q16_encode'
price_col = 'Q17_encode'

# Map the speed tiers using the mappings dictionary
data[speed_col] = data[speed_col].map(mappings[speed_col])

# Ensure that Q17_encode is mapped back to its numeric values for calculation purposes
data['Q17_mapped'] = data[price_col].map(lambda x: next(k for k, v in mappings[price_col].items() if v == x))

# Define the categorical type for Q16_encode to enforce the correct order
q16_order = CategoricalDtype(categories=[mappings[speed_col][i] for i in sorted(mappings[speed_col].keys())],
                             ordered=True)
data[speed_col] = data[speed_col].astype(q16_order)

# Get unique cluster labels and sort them
cluster_labels = sorted(data['Cluster_Label'].unique())

# Plot 1: Percentage of people per speed tier for all clusters
plt.figure(figsize=(12, 8))

# Loop over each cluster to calculate percentages and plot them
for cluster in cluster_labels:
    # Filter data for the current cluster
    cluster_data = data[data['Cluster_Label'] == cluster]

    # Group by speed tiers and calculate the percentage of people at each speed tier within the cluster
    cluster_size = len(cluster_data)
    percent_per_speed = (cluster_data.groupby(speed_col).size() / cluster_size) * 100

    # Plot the percentage values for the current cluster
    plt.plot(percent_per_speed.index, percent_per_speed.values, marker='o', label=f'Cluster {cluster}')

# Add titles and labels
plt.title('Percentage of People per Speed Tier for All Clusters')
plt.xlabel('Speed Tier (Q16_encode)')
plt.ylabel('Percentage of People')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.grid(True)
plt.legend()
# Save and display the plot
plt.savefig("Cluster_percentage_distribution_all_clusters.png")
plt.show()

# Plot 2: Average prices per speed tier for all clusters
plt.figure(figsize=(12, 8))

# Loop over each cluster in order to plot the mean on the same figure
for cluster in cluster_labels:
    # Filter data for the current cluster
    cluster_data = data[data['Cluster_Label'] == cluster]

    # Group by speed tiers and compute the mean
    avg_price_per_speed = cluster_data.groupby(speed_col)['Q17_mapped'].mean()

    # Map the mean prices back to the nearest price tier using the mappings dictionary
    avg_price_per_speed_mapped = avg_price_per_speed.map(lambda x: min(mappings[price_col], key=lambda k: abs(k - x)))

    # Order the y-axis based on the numeric keys in the mappings, but show the labels
    y_order = sorted(mappings[price_col].keys())
    y_labels = [mappings[price_col][key] for key in y_order]

    # Create a mapping from the numeric key back to the corresponding index in y_order
    avg_price_ordered = avg_price_per_speed_mapped.map(lambda x: y_order.index(x))

    # Plot the line plot with the mean for the current cluster
    plt.plot(avg_price_ordered.index, avg_price_ordered, marker='o', label=f'Cluster {cluster}')

# Add titles and labels
plt.title('Average Prices per Speed Tier for All Clusters')
plt.xlabel('Speed Tier (Q16_encode)')
plt.ylabel('Price Paid (Q17_encode)')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.yticks(ticks=range(len(y_labels)), labels=y_labels)  # Ensure y-axis is ordered and labeled correctly
plt.grid(True)
plt.legend()
# Save and display the plot
plt.savefig("Cluster_sensitivities_all_clusters.png")
plt.show()

# Grouped data
grouped_data = data.groupby(['Q16_encode', 'Q17_encode', 'Cluster_Label']).size().reset_index(name='Count')
# Calculate the percentage of people in each combination by Cluster_Label
grouped_data['Percentage'] = grouped_data.groupby('Cluster_Label')['Count'].transform(lambda x: (x / x.sum()) * 100)
# Display the resulting grouped data with percentages
print(grouped_data)

x=1
y=2