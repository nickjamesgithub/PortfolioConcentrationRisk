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
# Group by Anon_srvc_id and get the most recent MONTH_DT
most_recent = df.groupby('Anon_srvc_id')['MONTH_DT'].max().reset_index()
# Merge the most recent dates back with the original DataFrame to filter
df_final = pd.merge(df, most_recent, on=['Anon_srvc_id', 'MONTH_DT'])

# Define the columns to encode
columns_to_encode = ['tech_type', 'srvc_tenure_bn', 'plan_name', 'SPEED_TIER', 'SPEED_INCLUSION',
                     'bundle_type', 'value_bucket', 'RTC_Status', 'Modem_Model_Nam', 'Plan_Family',
                     'nbn_flag', 'nbn_tenure_typ', 'speed_prdct_name', 'Age_Bn', 'cust_tenure_bn',
                     'Has_PPM_fla', 'Has_MBB_fla', 'telstra_plus_stt', 'telstra_plus_tie',
                     'UP_DOWN_PLA', 'CHURN_ATTRIBUTED_RANKED', 'Tenure_Yrs', 'Fxd_Voice_Cn', 'Fxd_Brdbd_Cn',
                     'Mbl_Prpd_Cn', 'Mbl_Brdbd_Cn', 'FFT_Cnt', 'Bndl_Cnt', 'Pnsn_Cnt', 'Prrty_Cnt', 'TelstraTV_Cn', 'FBB_data_usg_MB_Mth',
                    'PPM_HH_Churn_cn', 'MBB_Churn_cn', 'PPM_HH_Starte',
                     'PPM_HH_Basi', 'PPM_HH_Essentia', 'PPM_HH_Premiu', 'PPM_HH_XL', 'PPM_MBB_Data_X']

# Churned customers
churned = df_final.loc[df_final["FBB_Churn_fla"] == 1].copy()
# Non-churned customers
active = df_final.loc[df_final["FBB_Churn_fla"] == 0].copy()

# To store feature distances
feature_distribution_distance = []

# Plotting normalized histograms using matplotlib
for column in columns_to_encode:
    # Remove rows with None, NaN, or other missing values in the current column
    churned_column = churned[[column]].dropna()
    active_column = active[[column]].dropna()

    if churned_column.empty or active_column.empty:
        print(f"Skipping {column} because it has no data after filtering.")
        continue

    try:
        plt.figure(figsize=(14, 7))
        # Initialize LabelEncoder for the current column
        encoder = LabelEncoder()
        # Fit the encoder on both churned and active data to ensure consistency
        combined_data = pd.concat([churned_column[column], active_column[column]], axis=0)
        encoder.fit(combined_data)
        churned_column[column + '_encoded'] = encoder.transform(churned_column[column])
        active_column[column + '_encoded'] = encoder.transform(active_column[column])

        # Compute histogram values without normalization
        churned_counts, bins = np.histogram(churned_column[column + '_encoded'], bins=len(encoder.classes_))
        active_counts, _ = np.histogram(active_column[column + '_encoded'], bins=len(encoder.classes_))

        # Convert counts to percentages
        churned_values = churned_counts / churned_counts.sum() * 100
        active_values = active_counts / active_counts.sum() * 100

        # Compute normalized inner product
        normalized_inner_product = np.dot(churned_values, active_values) / (np.sum(np.abs(churned_values)) * np.sum(np.abs(active_values)))
        feature_distribution_distance.append([column, normalized_inner_product])
        print("Churned total ", churned_values.sum())
        print("Active total ", active_values.sum())
        print("Feature Normalized inner product ", normalized_inner_product)

        bin_width = bins[1] - bins[0]
        bin_centers = bins[:-1] + bin_width / 2

        # Plot churned percentages
        plt.bar(bin_centers - bin_width / 4, churned_values, width=bin_width / 2, alpha=0.6, color='red', label='Churned')
        # Plot active percentages
        plt.bar(bin_centers + bin_width / 4, active_values, width=bin_width / 2, alpha=0.6, color='blue', label='Active')
        # Set labels and titles
        plt.xlabel(column)
        plt.ylabel('Percentage')
        plt.title(f'Normalized Distribution of {column} for Churned and Active Customers')

        # Replace x-axis numeric labels with the actual tech type
        labels = encoder.classes_
        plt.xticks(bin_centers, labels, rotation=90)
        plt.legend()
        # Adjust layout to make room for the labels
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.3)
        plt.savefig(f"Telstra_plot_{column}_customer_churn_vs_active.png")
        # plt.show()

        print('Iteration ', column)

    except Exception as e:
        print(f"Could not plot {column} due to error: {e}")

# Make a dataframe
fdd_df = pd.DataFrame(feature_distribution_distance)
fdd_df.to_csv(r'C:\Users\60848\OneDrive - Bain\Desktop\Telstra\Feature_Distribution_Distance.csv')

