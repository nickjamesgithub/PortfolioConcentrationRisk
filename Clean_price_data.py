import os
import pandas as pd

# Define the directory containing the CSV files
directory = r"C:\Users\60848\OneDrive - Bain\Desktop\Market_concentration_risk_paper\top_100_data_\data_"

# Create an empty list to store individual dataframes
dataframes = []

# Loop through each file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        # Read the CSV file into a dataframe
        df = pd.read_csv(os.path.join(directory, filename))
        # Extract the ticker from the filename (assuming the filename is the ticker)
        ticker = os.path.splitext(filename)[0]
        # Add a new column for the ticker
        df['ticker'] = ticker
        # Append the dataframe to the list
        dataframes.append(df)

# Concatenate all the dataframes into a single dataframe
combined_df = pd.concat(dataframes)
# Ensure no duplicate dates for each ticker by aggregating (taking the mean here)
combined_df = combined_df.groupby(['Date', 'ticker']).mean().reset_index()
# Pivot the dataframe to have tickers as columns and dates as rows
pivot_df = combined_df.pivot(index='Date', columns='ticker', values='Adj Close')

# Ensure dates are sorted
pivot_df.sort_index(inplace=True)

# Save the combined dataframe to a new CSV file
pivot_df.to_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Market_concentration_risk_paper\top_100_data_\combined_prices.csv")

print("Data successfully combined and saved.")