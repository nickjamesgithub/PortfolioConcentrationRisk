import pandas as pd

# Read in optimal weights
opt_weights = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Market_concentration_risk_paper\top_100_data_\Optimal_weights.csv")
opt_weights = opt_weights.iloc[:,1:]

x=1
y=2