#todo Import returns and market caps
#todo Compute matrix between all returns and all market caps on a monthly basis over time
#todo convert to affinity matrices and then
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy as sp
from scipy.signal import savgol_filter

matplotlib.use('TkAgg')
make_heatmap = False

# Read in ticker mapping
ticker_mapping = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Market_concentration_risk_paper\top_100_data_\Ticker_mapping.csv")
# Assuming the mapping file has columns 'Ticker' and 'Company Name' for old and new column names
# Create a dictionary from the mapping file
column_mapping_dict = dict(zip(ticker_mapping['Ticker'], ticker_mapping['Company Name']))
# Import market cap data
df_mcap = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Market_concentration_risk_paper\Market_Capitalisation_Data.csv")
# Melting the dataframe to have dates as rows
df_melted = pd.melt(df_mcap, id_vars=['Name'], var_name='Date', value_name='value')
# Convert Date column to datetime
df_melted['Date'] = pd.to_datetime(df_melted['Date'])
# Pivoting the dataframe to have company names as columns
df_pivoted = df_melted.pivot(index='Date', columns='Name', values='value')
df_pivoted.replace('Data Unavailable', np.nan, inplace=True)

# Convert columns to numeric, forcing errors to NaN
df_mcap_clean = df_pivoted.apply(pd.to_numeric, errors='coerce')
df_mcap_clean_ = df_mcap_clean.dropna(axis=1)

# Rename columns in market cap data using the mapping dictionary
df_mcap_clean_.rename(columns=column_mapping_dict, inplace=True)

# Read in price data
df_prices = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Market_concentration_risk_paper\top_100_data_\combined_prices.csv")
# Convert Date to datetime
df_prices['Date'] = pd.to_datetime(df_prices['Date'])
# Set Date as the index
df_prices.set_index('Date', inplace=True)
# Drop columns with any NaN values
df_prices = df_prices.dropna(axis=1)

# Make log returns
df_returns = np.log(df_prices) - np.log(df_prices.shift(1))
# Resample daily returns to monthly returns by summing the returns within each month
df_returns_monthly = df_returns.resample('M').sum()


# Map df_returns_monthly columns from tickers to company names
df_returns_monthly.columns = [column_mapping_dict.get(ticker, ticker) for ticker in df_returns_monthly.columns]
# Identify common columns
common_columns = df_returns_monthly.columns.intersection(df_mcap_clean_.columns)
# Filter and align both DataFrames to have the same columns
df_returns_monthly = df_returns_monthly[common_columns]
df_returns_monthly = df_returns_monthly.iloc[1:,:]
df_mcap_clean_ = df_mcap_clean_[common_columns]

# Now we loop over all months, form 2 distance matrices and study evolution
date_list = []
diff_norm_list = []
for i in range(len(df_returns_monthly)):
    # Get date
    date = df_returns_monthly.index[i]

    # Form distance matrices
    dist_matrix_returns_i = sp.spatial.distance_matrix(np.array(df_returns_monthly.iloc[i,:]).reshape(-1,1), np.array(df_returns_monthly.iloc[i,:]).reshape(-1,1))
    dist_matrix_mcap_i = sp.spatial.distance_matrix(np.array(df_mcap_clean_.iloc[i, :]).reshape(-1, 1), np.array(df_mcap_clean_.iloc[i, :]).reshape(-1, 1))

    # Form affinity matrices
    affinity_returns_i = 1 - dist_matrix_returns_i/np.max(dist_matrix_returns_i)
    affinity_mcap_i = 1 - dist_matrix_mcap_i/np.max(dist_matrix_mcap_i)

    # Matrix norm
    diff_norm = np.sum(np.abs(affinity_mcap_i - affinity_returns_i))

    # Append to lists
    date_list.append(date)
    diff_norm_list.append(diff_norm)

    # Print iteration
    print(date)

# Ensure the dates are in datetime format
date_list = pd.to_datetime(date_list)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(date_list, diff_norm_list, marker='o')
plt.plot(date_list, savgol_filter(diff_norm_list, 31, 2), color='red', alpha = 0.8)
# Set maximum number of tick labels to 7 and format them
ax = plt.gca()
ax.xaxis.set_major_locator(plt.MaxNLocator(7))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: pd.to_datetime(x).strftime('%b-%Y')))

# Set x-tick labels to month-year format
date_labels = [date.strftime('%b-%Y') for date in date_list]
ax.set_xticks(date_list[::len(date_list)//7])  # Adjust to have at most 7 labels
ax.set_xticklabels(date_labels[::len(date_labels)//7], rotation=90)

plt.xlabel('Date')
plt.ylabel('Differenced Matrix Norm')
plt.title('Differenced Matrix Norm vs Time')
plt.tight_layout()
plt.show()