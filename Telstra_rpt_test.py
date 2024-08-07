import pandas as pd

# Correct the file path
file = r'C:\Users\60848\OneDrive - Bain\Desktop\Telstra\Bain_6Aug24.rpt'

# Read the first line to determine column specifications
count = 1
for x in open(file, encoding='utf8'):
    cols = x.rstrip()
    count += 1
    if count > 2:
        break

# Calculate column specifications based on spaces between columns
colspecs = []
idx = 0
for c in cols.split(' '):
    n = len(c)
    colspecs.append((idx, idx + n))
    idx += 1 + n

# Read the fixed-width file into a DataFrame
df = pd.read_fwf(file, colspecs=colspecs, encoding='utf8', skiprows=[1])
# df.to_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Telstra\Bain_data.csv")
# Save the DataFrame to a Parquet file
df.to_parquet(r'C:\Users\60848\OneDrive - Bain\Desktop\Telstra\Bain_data.parquet')

# Display the first few rows to verify the contents
print(df.head())

# Churned customers
churned = df.loc[df["FBB_Churn_fla"]==1]
# Churned service IDs
service_ids = df.loc[df["FBB_Churn_fla"]==1]["Anon_srvc_id"]
test_churn_slice = df.loc[df["Anon_srvc_id"].isin(service_ids.values)]

