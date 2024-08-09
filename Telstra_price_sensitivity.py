import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

# Import data
df = pd.read_parquet(r'C:\Users\60848\OneDrive - Bain\Desktop\Telstra\Bain_data.parquet')
# Get the most recent data for each unique customer (churned/non-churned)
df['MONTH_DT'] = pd.to_datetime(df['MONTH_DT'])

# Slice anonymous service ID
slice = df.loc[df["Anon_srvc_id"]=="0x000F4D06769C83D5163A7D3E90B10A57C26A7A5A942483CEFDD713251E1360FB"][["MONTH_DT", "Tenure_Mnth", "FBB_Churn_fla", "FBB_data_usg_MB_Mth"]]

# Plot the frequency of the 'CHURN_ATTRIBUTED_RANK' column
plt.figure(figsize=(10, 6))
churn_attributed_rank_counts = df['CHURN_ATTRIBUTED_RANKED'].value_counts()
churn_attributed_rank_counts.plot(kind='bar', color='skyblue')
plt.xlabel('CHURN_ATTRIBUTED_RANK')
plt.ylabel('Frequency')
plt.title('Frequency of CHURN_ATTRIBUTED_RANK')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("CHURN_ATTRIBUTED_RANK_Frequency.png")
plt.show()

# Filter for churned customers
churned_df = df[df['FBB_Churn_fla'] == 1]
# Group by month and count churned customers
churned_per_month = churned_df.groupby(churned_df['MONTH_DT'].dt.to_period('M')).size()
# Calculate the number of price-sensitive churned customers per month
price_sensitive_churned_per_month = churned_df[churned_df['Price_Sensitiv'] == 'Y'].groupby(churned_df['MONTH_DT'].dt.to_period('M')).size()

# Create a DataFrame to hold both total and price-sensitive churned customers
churn_data = pd.DataFrame({
    'Total_Churned': churned_per_month,
    'Price_Sensitive_Churned': price_sensitive_churned_per_month
}).fillna(0)  # Fill NaN values with 0

# Calculate the percentage of price-sensitive churned customers
churn_data['Percentage_Price_Sensitive'] = (churn_data['Price_Sensitive_Churned'] / churn_data['Total_Churned']) * 100
# Group by month and count total customers per month
total_customers_per_month = df.groupby(df['MONTH_DT'].dt.to_period('M')).size()
# Calculate the percentage of customers that churned each month
percentage_churned_per_month = (churned_per_month / total_customers_per_month) * 100

# Plotting total churned customers per month
plt.figure(figsize=(10, 6))
churned_per_month.plot(kind='bar', color='skyblue')
plt.xlabel('Month')
plt.ylabel('Number of Churned Customers')
plt.title('Number of Churned Customers Per Month')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Churned_Customers_Per_Month.png")
plt.show()

# Plotting price-sensitive churned customers per month
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the total churned customers
bars = ax.bar(churn_data.index.astype(str), churn_data['Total_Churned'], color='skyblue', label='Total Churned Customers')

# Overlay the price-sensitive churned customers
for bar, price_sensitive, percentage in zip(bars, churn_data['Price_Sensitive_Churned'], churn_data['Percentage_Price_Sensitive']):
    height = bar.get_height()
    price_sensitive_height = (price_sensitive / churn_data['Total_Churned'].max()) * height
    ax.bar(bar.get_x(), price_sensitive_height, width=bar.get_width(), color='orange', label='Price Sensitive' if bar.get_x() == bars[0].get_x() else "")
    ax.text(bar.get_x() + bar.get_width() / 2, height, f'{percentage:.1f}%', ha='center', va='bottom', fontsize=9, color='black')

# Add labels and title
ax.set_xlabel('Month')
ax.set_ylabel('Number of Churned Customers')
ax.set_title('Number of Churned Customers Per Month and Price Sensitivity')
ax.legend()

# Rotate x-axis labels
plt.xticks(rotation=45)
# Save and show the plot
plt.tight_layout()
plt.savefig("Price_Sensitivity_Churned_Customers_Per_Month.png")
plt.show()

# Plotting percentage of total customers that churned each month
plt.figure(figsize=(10, 6))
percentage_churned_per_month.plot(kind='bar', color='skyblue')
plt.xlabel('Month')
plt.ylabel('Percentage of Total Customers Churned')
plt.title('Percentage of Total Customers That Churned Per Month')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Percentage_Churned_Customers_Per_Month.png")
plt.show()
