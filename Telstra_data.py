import pandas as pd
import numpy as np
import re
from fuzzywuzzy import process
import tqdm

# Import data
data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Telstra\Bain_data3.csv")

# Define a function to split the data based on multiple delimiters
def split_data(cell):
    if isinstance(cell, str):
        return re.split(r'\s+|,|;', cell)
    else:
        return [cell]  # Return the cell as a single-element list if it's not a string

# Apply the function to the first column and expand into multiple columns
df_split = data.iloc[:, 0].apply(split_data).apply(pd.Series)

# Define the reference row (e.g., the first row)
reference_row = df_split.iloc[0].dropna().tolist()

# Function to align a row to the reference row with fuzzy matching and adjustments
def align_row(row, reference):
    row = [item for item in row if pd.notnull(item)]  # Remove NaN values
    aligned_row = [''] * len(reference)  # Initialize aligned row with empty strings
    matched_indices = []
    for item in row:
        match = process.extractOne(item, reference)
        if match and match[1] > 75:  # Use a threshold for matching confidence
            idx = reference.index(match[0])
            aligned_row[idx] = item
            matched_indices.append(idx)
    for i, item in enumerate(row):
        if i not in matched_indices:
            for j in range(len(aligned_row)):
                if aligned_row[j] == '':
                    aligned_row[j] = item
                    break
    return aligned_row

# Align all rows to the reference row with a progress bar
aligned_data = []
for idx, row in enumerate(tqdm.tqdm(df_split.itertuples(index=False), total=len(df_split))):
    aligned_data.append(align_row(row, reference_row))

aligned_df = pd.DataFrame(aligned_data, columns=reference_row)

# If there are additional columns in the original DataFrame, concatenate them
if data.shape[1] > 1:
    df_final = pd.concat([aligned_df, data.iloc[:, 1:].reset_index(drop=True)], axis=1)
else:
    df_final = aligned_df

# Save the cleaned data to a new CSV file
df_final.to_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Telstra\cleaned_Bain_data3.csv", index=False)
