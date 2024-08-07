import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

# Turn feature engineering label on
feature_engineering = True

# Import data
df = pd.read_parquet(r'C:\Users\60848\OneDrive - Bain\Desktop\Telstra\Bain_data.parquet')
# Get the most recent data for each unique customer (churned/non-churned)
df['MONTH_DT'] = pd.to_datetime(df['MONTH_DT'])
# Group by Anon_srvc_id and get the most recent MONTH_DT
most_recent = df.groupby('Anon_srvc_id')['MONTH_DT'].max().reset_index()
# Merge the most recent dates back with the original DataFrame to filter
df_final = pd.merge(df, most_recent, on=['Anon_srvc_id', 'MONTH_DT'])
# Remove last 3 rows of junk data
df_slice = df_final.iloc[:-3,:]

# Define the columns to encode
columns_to_encode = ['tech_type', 'srvc_tenure_bn', 'plan_name', 'SPEED_TIER', 'SPEED_INCLUSION',
                     'bundle_type', 'value_bucket', 'RTC_Status', 'Modem_Model_Nam', 'Plan_Family',
                     'nbn_flag', 'nbn_tenure_typ', 'speed_prdct_name', 'Age_Bn', 'cust_tenure_bn',
                     'Has_PPM_fla', 'Has_MBB_fla', 'telstra_plus_stt', 'telstra_plus_tie',
                     'UP_DOWN_PLA', 'CHURN_ATTRIBUTED_RANKED', 'Tenure_Yrs', 'Fxd_Voice_Cn', 'Fxd_Brdbd_Cn',
                     'Mbl_Prpd_Cn', 'Mbl_Brdbd_Cn', 'FFT_Cnt', 'Bndl_Cnt', 'Pnsn_Cnt', 'Prrty_Cnt', 'TelstraTV_Cn', 'FBB_data_usg_MB_Mth',
                     'PPM_HH_Churn_cn', 'MBB_Churn_cn', 'PPM_HH_Starte',
                     'PPM_HH_Basi', 'PPM_HH_Essentia', 'PPM_HH_Premiu', 'PPM_HH_XL', 'PPM_MBB_Data_X']

# Define the target variable
target = 'FBB_Churn_fla'

# Separate features and target variable
X = df_slice[columns_to_encode]
y = df_slice[target]

# Encode categorical features
label_encoders = {}
for column in columns_to_encode:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate the model
y_pred = rf.predict(X_test)
print(f'Accuracy: {rf.score(X_test, y_test)}')

# Extract feature importances
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot the feature importances
plt.figure(figsize=(14, 7))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [columns_to_encode[i] for i in indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.tight_layout()
plt.savefig("Feature_importance_churn_model")
plt.show()
