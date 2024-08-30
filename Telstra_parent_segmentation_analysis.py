import pandas as pd
import numpy as np
from scipy.stats import mode

# Define a function to return the mode; handling multiple modes by returning the first one
def get_mode(series):
    mode_values = series.mode()
    if len(mode_values) > 0:
        return mode_values.iloc[0]
    else:
        return np.nan  # Return NaN if there's no mode

# Read in data
data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Telstra\Data_with_cluster_projections.csv")

# Define which prefixes to apply mean and which to apply mode
mean_prefixes = ["Q1_encode", "Q4_encode", "Q8_encode", "Q14_encode", "Q16_encode", "Q17_encode", "Q23_encode", "Q30_encode",
                 "Q31_encode", "Q35_encode", "Q41_encode", "Q42_encode"]
mode_prefixes = ["Q15_encode"]

# Aggregate functions based on column prefixes
agg_dict = {}

# For mean_prefixes, apply mean
for prefix in mean_prefixes:
    cols = [col for col in data.columns if col.startswith(prefix)]
    if cols:
        print(f"Columns matched for prefix '{prefix}': {cols}")
    else:
        print(f"No columns matched for prefix '{prefix}'")
    agg_dict.update({col: 'mean' for col in cols})

# For mode_prefixes, apply mode
for prefix in mode_prefixes:
    cols = [col for col in data.columns if col.startswith(prefix)]
    if cols:
        print(f"Columns matched for prefix '{prefix}': {cols}")
    else:
        print(f"No columns matched for prefix '{prefix}'")
    agg_dict.update({col: get_mode for col in cols})

# Group by 'Cluster_Label' and perform the aggregation
grouped_df = data.groupby('Cluster_Label').agg(agg_dict).reset_index()

# Display the grouped DataFrame with aggregated results
print(grouped_df)

x=1
y=2