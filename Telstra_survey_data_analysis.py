import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from Utilities import dendrogram_plot_test
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kendalltau
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids
import gower
from sklearn.preprocessing import LabelEncoder


# Read in data
# data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Telstra\Survey_data.csv")
data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Telstra\survey_responses_240909.csv")
data_length = len(data)
make_plots = True

def kendall_tau_distance_matrix(data):
    def kendall_tau_distance(u, v):
        tau, _ = kendalltau(u, v)
        return 1 - tau
    # Use pdist to compute the pairwise Kendall Tau distances
    distance_vector = pdist(data, metric=kendall_tau_distance)
    # Convert the condensed distance vector to a squareform matrix
    distance_matrix = squareform(distance_vector)
    return distance_matrix

def l1_distance_matrix(series):
    series_np = series.to_numpy()  # Convert the pandas Series to a NumPy array
    return np.nan_to_num(np.abs(series_np[:, np.newaxis] - series_np))

# Step 1: Define a function to calculate the L1 (Manhattan) distance between two vectors
def l1_distance_matrix_q14(df):
    df_np = df.to_numpy()  # Convert the DataFrame to a NumPy array
    return np.sum(np.abs(df_np[:, np.newaxis] - df_np), axis=2)

def cosine_similarity_matrix(data):
    # Convert DataFrame to NumPy array
    data_np = data.to_numpy()
    norms = np.linalg.norm(data_np, axis=1)
    return np.nan_to_num(np.dot(data_np, data_np.T) / np.outer(norms, norms))

def convert_to_affinity(*matrices):
    return [(1 - matrix / np.max(matrix)) for matrix in matrices]

# Sum over all matrices in the dictionary
def sum_affinity_matrices(affinity_dict):
    # Get the list of all matrices from the dictionary
    matrices = list(affinity_dict.values())
    # Sum the matrices element-wise
    total_matrix = np.sum(matrices, axis=0)
    return total_matrix

def plot_mad_affinity(affinity_dict, x_labels):
    # Calculate the Median Absolute Deviation (MAD) for each affinity matrix
    mad_values = [np.median(np.abs(matrix - np.median(matrix))) for matrix in affinity_dict.values()]

    # Create the line plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_labels, mad_values, marker='o', linestyle='-', color='b')
    plt.title('MAD of Affinity Matrices')
    plt.xlabel('Affinity Matrix')
    plt.ylabel('Median Absolute Deviation (MAD)')
    plt.grid(True)
    plt.xticks(rotation=45)  # Rotate labels for better readability if needed
    plt.savefig("Questions_separability_MAD")
    plt.show()

# Feature engineer all the data
# Q1-Q4 - Replace categorical data with numeric values
# Q1
q1_encoded = data["Q1"].replace(['18-24', "25-34", '35-44', '45-54', '55-64', '65-74', '75 and above'], [1, 2, 3, 4, 5, 6, 7])
data["Q1_encode"] = q1_encoded
# Q2
q2_encoded = data["Q2"]
data["Q2_encode"] = q2_encoded
# Q3
q3_encoded = data["Q3"].replace(["Male", "Female", "Non-binary"], [0,1,2])
data["Q3_encode"] = q3_encoded
# Q4
q4_encoded = data["Q4"].replace(['Less than $15,000', '$15,000 to $29,999', '$30,000 to $44,999','$45,000 to $59,999',
'$60,000 to $79,999','$80,000 to $99,999','$100,000 to $119,999','$120,000 to $149,999', '$150,000 to $199,999','$200,000 or more'],
                                       [1,2,3,4,5,6,7,8,9,10])
data["Q4_encode"] = q4_encoded

# Q5 - Cosine similarity
# Identify columns that start with 'Q5' exactly
q5_columns = [col for col in data.columns if col.startswith('Q5') and col[2:3].isdigit() == False]
# Create a new DataFrame with the renamed columns
q5_encoded = data[q5_columns].copy()
q5_encoded.columns = [col.replace('Q5', 'Q5_encode') for col in q5_encoded.columns]
# Append the newly renamed columns to the far right of the original DataFrame
data = pd.concat([data, q5_encoded], axis=1)

# Q6 - one hot encode
q6_encoded = pd.get_dummies(data["Q6"], prefix='Q6_encode').astype(int)
data = data.join(q6_encoded)
# Q7 - one hot encode
q7_encoded = pd.get_dummies(data["Q7"], prefix='Q7_encode').astype(int)
data = data.join(q7_encoded)
# Q8 - L1 distance
q8_encoded = data["Q8"].replace(["I need to shop within a set budget for my home internet services",
                                 "I need the lowest cost service because of my budget",
                                 "I choose based on best value (what I get for my dollars)",
                                 "I choose the most competitive price for the internet solution that meets my requirements",
                                 "I want something that is more premium, price is less important",
                                 "I don't want to spend time thinking about it, I'm happy to pay for something that just works",
                                 "I am happy to pay for the most innovative offering",
                                 "I need the latest/best technology for my home internet service, I don’t care how much it costs"], [1,1,2,2,3,3,4,4])

data["Q8_encode"] = q8_encoded

# Q9 - one hot encode
q9_encoded = pd.get_dummies(data["Q9"], prefix='Q9_encode').astype(int)
data = data.join(q9_encoded)

# Q11 - Cosine similarity
# Identify columns that start with 'Q11' exactly
q11_columns = [col for col in data.columns if col.startswith('Q11') and col[3:4].isdigit() == False]
# Create a new DataFrame with the renamed columns
q11_encoded = data[q11_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
q11_encoded.columns = [col.replace('Q11', 'Q11_encode') for col in q11_encoded.columns]
# Append the newly renamed columns to the far right of the original DataFrame
data = pd.concat([data, q11_encoded], axis=1)

# Q12 - Cosine similarity
q12_columns = [col for col in data.columns if col.startswith('Q12') and col[3:4].isdigit() == False]
# Create a new DataFrame with the renamed columns
q12_encoded = data[q12_columns].copy().fillna(0)
q12_encoded.columns = [col.replace('Q12', 'Q12_encode') for col in q12_encoded.columns]
# Append the newly renamed columns to the far right of the original DataFrame
data = pd.concat([data, q12_encoded], axis=1)

# Q13 - one hot encode
q13_encoded = pd.get_dummies(data["Q13"], prefix='Q13_encode').astype(int)
data = data.join(q13_encoded)

# Q14 - L1 distance
# Step 1: Identify and select the relevant Q14 columns
q14_columns = [col for col in data.columns if col.startswith('Q14') and col[3:4].isdigit() == False]
# Step 2: Create a new DataFrame with the selected columns, replacing NaN values with 0
q14_encoded = data[q14_columns].copy().fillna(0)
# Step 3: Rename the columns to have 'Q14_encode' instead of 'Q14'
q14_encoded.columns = [col.replace('Q14', 'Q14_encode') for col in q14_encoded.columns]
# Step 4: Define the mapping of old values to new values
value_mapping = {0: 0, 1: 100, 2: 40, 3: 20, 4: 10, 5: 5}
# Step 5: Replace the values in the q14_encoded DataFrame using the mapping
q14_encoded = q14_encoded.applymap(lambda x: value_mapping.get(x, x))
# Step 6: Append the updated q14_encoded columns to the original DataFrame
data = pd.concat([data, q14_encoded], axis=1)
q14_encoded_transposed = q14_encoded.T

# Q15 - Cosine similarity
# Identify columns that start with 'Q15' exactly
q15_columns = [col for col in data.columns if col.startswith('Q15') and col[3:4].isdigit() == False]
# Create a new DataFrame with the renamed columns
q15_encoded = data[q15_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
q15_encoded.columns = [col.replace('Q15', 'Q15_encode') for col in q15_encoded.columns]
# Append the newly renamed columns to the far right of the original DataFrame
data = pd.concat([data, q15_encoded], axis=1)

# Q16 - L1 Distance
q16_encoded = data["Q16"].replace(['I don’t know',
'4G wireless broadband (fixed wireless)','5G wireless broadband (fixed wireless)',
       '12 mbps', '25 mbps', '50 mbps', '100 mbps', '250 mbps', '500 mbps',
       '750 mbps', '1,000 mbps'],[1,2,3,4,5,6,7,8,9,10,11])
data["Q16_encode"] = q16_encoded

# Q17 - L1 Distance
q17_encoded = data["Q17"].replace(['I don’t know','Less than $50','$50 - $54.99','$55 - $59.99','$60 - $64.99','$65 - $69.99',
'$70 - $74.99', '$75 - $79.99', '$80 - $84.99','$85 - $89.99', '$90 - $94.99','$95 - $99.99','$100 - $104.99', '$105 - $109.99',
'$110 - $114.99', '$115 - $119.99','$120 - $124.99','$125 - $129.99', '$130 - $134.99', '$135 - $139.99','$140 - $144.99','$145 - $149.99', '$150 or more'],[1,2,3,4,5,6,7,8,9,10,11,12,
                                                                                                                                              13,14,15,16,17,18,19,20,21,22,23])
data["Q17_encode"] = q17_encoded

# Q18 - Cosine similarity
# Identify columns that start with 'Q18' exactly
q18_columns = [col for col in data.columns if col.startswith('Q18') and col[3:4].isdigit() == False]
# Create a new DataFrame with the renamed columns
q18_encoded = data[q18_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
q18_encoded.columns = [col.replace('Q18', 'Q18_encode') for col in q18_encoded.columns]
# Append the newly renamed columns to the far right of the original DataFrame
data = pd.concat([data, q18_encoded], axis=1)

# Q19 - one hot encode
q19_encoded = pd.get_dummies(data["Q19"], prefix='Q19_encode').astype(int)
data = data.join(q19_encoded)

# Q21 - Cosine similarity
# Identify columns that start with 'Q21' exactly
q21_columns = [col for col in data.columns if col.startswith('Q21') and col[3:4].isdigit() == False]
# Create a new DataFrame with the renamed columns
q21_encoded = data[q21_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
q21_encoded.columns = [col.replace('Q21', 'Q21_encode') for col in q21_encoded.columns]
# Append the newly renamed columns to the far right of the original DataFrame
data = pd.concat([data, q21_encoded], axis=1)

# Q22 - one hot encode
q22_encoded = pd.get_dummies(data["Q22"], prefix='Q22_encode').astype(int)
data = data.join(q22_encoded)

# Q23
q23_encoded = data["Q23"]
data["Q23_encode"] = q23_encoded

# Q24 - Cosine similarity
# Identify columns that start with 'Q24' exactly
q24_columns = [col for col in data.columns if col.startswith('Q24') and col[3:4].isdigit() == False]
# Create a new DataFrame with the renamed columns
q24_encoded = data[q24_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
q24_encoded.columns = [col.replace('Q24', 'Q24_encode') for col in q24_encoded.columns]
# Append the newly renamed columns to the far right of the original DataFrame
data = pd.concat([data, q24_encoded], axis=1)

# Q25 - one hot encode
q25_encoded = pd.get_dummies(data["Q25"], prefix='Q25_encode').astype(int)
data = data.join(q25_encoded)

# Q27 - one hot encode
q27_encoded = pd.get_dummies(data["Q27"], prefix='Q27_encode').astype(int)
data = data.join(q27_encoded)

# Q29 - one hot encode
q29_encoded = pd.get_dummies(data["Q29"], prefix='Q29_encode').astype(int)
data = data.join(q29_encoded)

# Q30
q30_encoded = data["Q30"].replace(['Less than 6 months ', '6-12 months ', '12-24 months', '2-4 years',
       'More than 4 years', "5-7 years", "8-10 years", "Over 10 years"], [0,1,2,3,4,5,6,7])
data["Q30_encode"] = q30_encoded

# Q31
q31_encoded = data["Q31"].replace(['Never', 'Over 10 years', '8-10 years ago', '5-7 years ago', 'More than 4 years ago', '2-4 years ago',
                                   '12-24 months ago','6-12 months ago', 'In the last 6 months'], [0,1,2,3,4,5,6,7,8])
data["Q31_encode"] = q31_encoded

# Q32 - Cosine similarity
# Identify columns that start with 'Qew' exactly
q32_columns = [col for col in data.columns if col.startswith('Q32') and col[3:4].isdigit() == False]
# Create a new DataFrame with the renamed columns
q32_encoded = data[q32_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
q32_encoded.columns = [col.replace('Q32', 'Q32_encode') for col in q32_encoded.columns]
# Append the newly renamed columns to the far right of the original DataFrame
data = pd.concat([data, q32_encoded], axis=1)

# Q33 - one hot encode
q33_encoded = pd.get_dummies(data["Q33"], prefix='Q33_encode').astype(int)
data = data.join(q33_encoded)

# Q34 - Cosine similarity
# Identify columns that start with 'Q34' exactly
q34_columns = [col for col in data.columns if col.startswith('Q34') and col[3:4].isdigit() == False]
# Create a new DataFrame with the renamed columns
q34_encoded = data[q34_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
q34_encoded.columns = [col.replace('Q34', 'Q34_encode') for col in q34_encoded.columns]
# Append the newly renamed columns to the far right of the original DataFrame
data = pd.concat([data, q34_encoded], axis=1)

# Q35
q35_encoded = data["Q35"].fillna(0)
data["Q35_encode"] = q35_encoded

# Q36 - Cosine similarity
# Identify columns that start with 'Q36' exactly
q36_columns = [col for col in data.columns if col.startswith('Q36') and col[3:4].isdigit() == False]
# Create a new DataFrame with the renamed columns
q36_encoded = data[q36_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
q36_encoded.columns = [col.replace('Q36', 'Q36_encode') for col in q36_encoded.columns]
# Append the newly renamed columns to the far right of the original DataFrame
data = pd.concat([data, q36_encoded], axis=1)

# Q37 - Cosine similarity
# Identify columns that start with 'Q37' exactly
q37_columns = [col for col in data.columns if col.startswith('Q37') and col[3:4].isdigit() == False]
# Create a new DataFrame with the renamed columns
q37_encoded = data[q37_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
q37_encoded.columns = [col.replace('Q37', 'Q37_encode') for col in q37_encoded.columns]
# Append the newly renamed columns to the far right of the original DataFrame
data = pd.concat([data, q37_encoded], axis=1)

# Q38 - Cosine similarity
# Identify columns that start with 'Q38' exactly
q38_columns = [col for col in data.columns if col.startswith('Q38') and col[3:4].isdigit() == False]
# Create a new DataFrame with the renamed columns
q38_encoded = data[q38_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
q38_encoded.columns = [col.replace('Q38', 'Q38_encode') for col in q38_encoded.columns]
# Append the newly renamed columns to the far right of the original DataFrame
data = pd.concat([data, q38_encoded], axis=1)

# Q41 - L1 Distance
q41_encoded = data["Q41"].replace([np.nan, 'I don’t know', '12 mbps', '25 mbps', '50 mbps', '100 mbps', '250 mbps',
        '500 mbps', '750 mbps', '1,000 mbps', ],[0,1,2,3,4,5,6,7,8,9])
data["Q41_encode"] = q41_encoded

# Q42 - numeric data
q42_encoded = data["Q42"].fillna(0)
data["Q42_encode"] = q42_encoded

# Q43 - one hot encode
q43_encoded = pd.get_dummies(data["Q43"], prefix='Q43_encode').astype(int)
data = data.join(q43_encoded)

# Q44 - one hot encode
q44_encoded = pd.get_dummies(data["Q44"], prefix='Q44_encode').astype(int)
data = data.join(q44_encoded)

# Compute distance matrices
# L1 computation
matrix_q1 = l1_distance_matrix(q1_encoded)
matrix_q3 = l1_distance_matrix(q3_encoded)
matrix_q4 = l1_distance_matrix(q4_encoded)
matrix_q8 = l1_distance_matrix(q8_encoded)
matrix_q14 = l1_distance_matrix_q14(q14_encoded)
matrix_q16 = l1_distance_matrix(q16_encoded)
matrix_q17 = l1_distance_matrix(q17_encoded)
matrix_q23 = l1_distance_matrix(q23_encoded)
matrix_q30 = l1_distance_matrix(q30_encoded)
matrix_q31 = l1_distance_matrix(q31_encoded)
matrix_q35 = l1_distance_matrix(q35_encoded)
matrix_q41 = l1_distance_matrix(q41_encoded)
matrix_q42 = l1_distance_matrix(q42_encoded)

# Cosine similarity
matrix_q5 = cosine_similarity_matrix(q5_encoded)
matrix_q6 = cosine_similarity_matrix(q6_encoded)
matrix_q7 = cosine_similarity_matrix(q7_encoded)
matrix_q9 = cosine_similarity_matrix(q9_encoded)
matrix_q11 = cosine_similarity_matrix(q11_encoded)
matrix_q12 = cosine_similarity_matrix(q12_encoded)
matrix_q13 = cosine_similarity_matrix(q13_encoded)

matrix_q15 = cosine_similarity_matrix(q15_encoded)
matrix_q18 = cosine_similarity_matrix(q18_encoded)
matrix_q19 = cosine_similarity_matrix(q19_encoded)
matrix_q21 = cosine_similarity_matrix(q21_encoded)
matrix_q22 = cosine_similarity_matrix(q22_encoded)
matrix_q24 = cosine_similarity_matrix(q24_encoded)
matrix_q25 = cosine_similarity_matrix(q25_encoded)
matrix_q27 = cosine_similarity_matrix(q27_encoded)
matrix_q29 = cosine_similarity_matrix(q29_encoded)
matrix_q32 = cosine_similarity_matrix(q32_encoded)
matrix_q33 = cosine_similarity_matrix(q33_encoded)
matrix_q34 = cosine_similarity_matrix(q34_encoded)
matrix_q36 = cosine_similarity_matrix(q36_encoded)
matrix_q37 = cosine_similarity_matrix(q37_encoded)
matrix_q38 = cosine_similarity_matrix(q38_encoded)
matrix_q43 = cosine_similarity_matrix(q43_encoded)
matrix_q44 = cosine_similarity_matrix(q44_encoded)

# Compute affinity matrices
distance_matrices = convert_to_affinity(matrix_q1, matrix_q3, matrix_q4,
                                   matrix_q5, matrix_q6, matrix_q7,
                                   matrix_q8, matrix_q9, matrix_q11,
                                   matrix_q12, matrix_q13, matrix_q14, matrix_q15,
                                   matrix_q16, matrix_q17, matrix_q18, matrix_q19,
                                   matrix_q21, matrix_q22, matrix_q23, matrix_q24,
                                   matrix_q25, matrix_q27, matrix_q29, matrix_q30,
                                   matrix_q31,matrix_q32,matrix_q33, matrix_q34, matrix_q35,
                                   matrix_q36, matrix_q37, matrix_q38, matrix_q41, matrix_q42,
                                        matrix_q43, matrix_q44)

select_matrices = convert_to_affinity(matrix_q1, matrix_q4,
                                   matrix_q5, matrix_q6, matrix_q8, matrix_q11,
                                    matrix_q14, matrix_q15,
                                   matrix_q16, matrix_q17, matrix_q23, matrix_q30,
                                   matrix_q31,matrix_q35, matrix_q41, matrix_q42)

labels = ["Q1", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q11",
          "Q12", "Q13", "Q14", "Q15", "Q16", "Q17", "Q18", "Q19",
          "Q21", "Q22", "Q23", "Q24", "Q25", "Q27", "Q29", "Q30",
          "Q31", "Q32", "Q33", "Q34", "Q35", "Q36", "Q37", "Q38",
          "Q41", "Q42", "Q43", "Q44"]

# If you want to store them in a dictionary with keys:
affinity_dict = {f"aff_matrix_q{i+1}": aff for i, aff in enumerate(distance_matrices)}
select_affinity_dict = {f"aff_matrix_q{i+1}": aff for i, aff in enumerate(select_matrices)}

# Example usage
plot_mad_affinity(affinity_dict, labels)

def plot_and_save_affinity_matrices(affinity_dict, labels, output_dir="affinity_heatmaps"):
    # Ensure the output directory exists
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over the matrices in the dictionary and their corresponding labels
    for label, (key, matrix) in zip(labels, affinity_dict.items()):
        plt.figure(figsize=(8, 6))
        plt.matshow(matrix, cmap="viridis")
        plt.title(f'Affinity Matrix: {label}')
        plt.xlabel('Index')
        plt.ylabel('Index')

        # Save the plot to the output directory with the label as the filename
        plt.savefig(f"affinity_matrix_{label}.png")
        plt.close()

# # Call the function to plot and save the heatmaps
# plot_and_save_affinity_matrices(affinity_dict, labels)

segment_matrices = [matrix_q1, matrix_q8, matrix_q11, matrix_q14, matrix_q15, matrix_q16, matrix_q17,
                     matrix_q23, matrix_q30, matrix_q31]

if make_plots:
    # Innate behaviour (Segments) - 4
    dendrogram_plot_test(matrix_q8, "Q8_innate", "Survey", np.linspace(1,data_length,data_length))
    # KPCs - 5
    dendrogram_plot_test(matrix_q14, "Q14_KPC", "Survey", np.linspace(1,data_length,data_length))
    # Activity - 5
    dendrogram_plot_test(matrix_q15, "Q15_Activity", "Activity", np.linspace(1,data_length,data_length))
    # Speed - 5
    dendrogram_plot_test(matrix_q16, "Q16_speed", "Survey", np.linspace(1, data_length, data_length))
    # Budget - 5
    dendrogram_plot_test(matrix_q17, "Q17_budget", "Survey", np.linspace(1,data_length,data_length))
    # Switching
    dendrogram_plot_test(matrix_q31, "Q31_Switching", "Survey", np.linspace(1,data_length,data_length))


# Number of clusters for each k-means
k_q8_segment = 4  # (Innate behaviour)
k_q14_kpc = 4  # (KPC)
k_q15_activity = 4  # (activity)
k_q16_requirements = 3  # (Speed)
k_q17_budget = 3  # (Budget)
k_q31_switching = 3  # Switching propensity

# Perform k-means clustering on matrix_q8
kmeans_q8 = KMeans(n_clusters=k_q8_segment, random_state=42).fit(matrix_q8)
labels_q8 = kmeans_q8.labels_
# K means - KPC
kmeans_q14 = KMeans(n_clusters=k_q14_kpc, random_state=42).fit(matrix_q14)
labels_q14 = kmeans_q14.labels_
# K means
kmeans_q15 = KMeans(n_clusters=k_q15_activity, random_state=42).fit(matrix_q15)
labels_q15 = kmeans_q15.labels_
# Perform k-means clustering on matrix_q16
kmeans_q16 = KMeans(n_clusters=k_q16_requirements, random_state=42).fit(matrix_q16)
labels_q16 = kmeans_q16.labels_
# Perform k-means clustering on matrix_q17
kmeans_q17 = KMeans(n_clusters=k_q17_budget, random_state=42).fit(matrix_q17)
labels_q17 = kmeans_q17.labels_
# Perform k-means clustering on matrix_q31
kmeans_q31 = KMeans(n_clusters=k_q31_switching, random_state=42).fit(matrix_q31)
labels_q31 = kmeans_q31.labels_

# Append labels to data
data["Innate_behaviour_cluster"] = labels_q8
data["KPC_cluster"] = labels_q14
data["Activity_cluster"] = labels_q15
data["Requirements_cluster"] = labels_q16
data["Budget_cluster"] = labels_q17
data["Switching_cluster"] = labels_q31

# Write clusters to master file
data.to_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Telstra\Data_clusters_engineered.csv")

# Group the DataFrame by the 'KPC' column
grouped_data = data.groupby(["KPC_cluster", 'Budget_cluster', 'Switching_cluster'])

# Specify the columns you want to compute the average of
columns_of_interest = [
    'Q14_encode: Price',
 'Q14_encode: Brand',
 'Q14_encode: Speed',
 'Q14_encode: Amount of data included',
 'Q14_encode: Installation support',
 'Q14_encode: Overseas-based customer support & service',
 'Q14_encode: Australia-based customer support & service',
 'Q14_encode: Home phone line included',
 'Q14_encode: <Entertainment offers>',
 'Q14_encode: <Modem brand>  ',
 'Q14_encode: 4G/5G mobile backup on the modem ',
 'Q14_encode: Ability to choose what’s included in my plan / build a bundle',
 'Q14_encode: Cutting edge technology']

agg_functions = {col: 'mean' for col in columns_of_interest}
agg_functions['Q1'] = lambda x: x.mode()[0] if not x.mode().empty else None
agg_functions['Q4'] = lambda x: x.mode()[0] if not x.mode().empty else None
agg_functions['Q6'] = lambda x: x.mode()[0] if not x.mode().empty else None
agg_functions['Q7'] = lambda x: x.mode()[0] if not x.mode().empty else None
agg_functions['Q16'] = lambda x: x.mode()[0] if not x.mode().empty else None
agg_functions['Q17'] = lambda x: x.mode()[0] if not x.mode().empty else None
agg_functions['Q31'] = lambda x: x.mode()[0] if not x.mode().empty else None
customer_counts = grouped_data.size().reset_index(name='Customer_Count')
# Apply the aggregation functions to get the mean and mode as specified
summary_table = grouped_data.agg(agg_functions).reset_index()
# Merge the customer count into the summary table
summary_table = pd.merge(summary_table, customer_counts, on=["KPC_cluster", 'Budget_cluster', 'Switching_cluster'])

# List of selected features to include in the clustering
selected_features = [
    'Q14_encode: Price',
    'Q14_encode: Brand',
    'Q31',
    # Add any other features you want to include
]

# Step 1: Filter the data based on selected features
data_filtered = data[selected_features].copy()

# # Handle missing values
# # For numerical columns, fill NaN with the median of those numerical columns
# data_numeric = data_filtered.select_dtypes(include=[np.number]).fillna(data_filtered.select_dtypes(include=[np.number]).median())
# # For categorical columns, fill NaN with the mode (most frequent value) of those categorical columns
# data_categorical = data_filtered.select_dtypes(include=[object]).apply(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "Unknown"))
# # Combine back into a single DataFrame
# data_filled = pd.concat([data_numeric, data_categorical], axis=1)
# # Step 2: Compute the Gower distance matrix using all customers in 'data'
# gower_dist_matrix = gower.gower_matrix(data_filled)
# # Step 3: Apply K-Medoids clustering
# n_clusters = 4  # Adjust this based on your needs
# kmedoids = KMedoids(n_clusters=n_clusters, metric='precomputed', random_state=42)
# kmedoids.fit(gower_dist_matrix)
# # Step 4: Append the cluster labels to the original DataFrame 'data'
# data['Cluster_3d'] = kmedoids.labels_
#
# # Step 1: Encode the categorical 'Q31' column
# le = LabelEncoder()
# summary_table['Q31_encode: Switching'] = le.fit_transform(summary_table['Q31'])
# # Step 2: Create a 3D scatter plot
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# # Plotting the data
# scatter = ax.scatter(
#     summary_table['Q14_encode: Price'],
#     summary_table['Q14_encode: Brand'],
#     summary_table['Q31_encode: Switching'],  # Use the encoded Q31 values for the z-axis
#     c=summary_table['Cluster_3d'],  # Color by cluster
#     cmap='viridis',  # Colormap, change as needed
#     s=50,  # Size of the markers
#     alpha=0.7  # Transparency of markers
# )
# # Setting axis labels
# ax.set_xlabel('Price')
# ax.set_ylabel('Brand')
# ax.set_zlabel('Q31_encode: Switching')
# # Adding a color bar to explain the cluster colors
# colorbar = plt.colorbar(scatter, ax=ax)
# colorbar.set_label('Cluster_3d')
# # Adding a title
# ax.set_title('3D Scatter Plot of Price, Brand, and Q31 by Cluster')
# # Show the plot
# plt.show()



# initialise distance matrix
consistency_matrix = np.zeros(((len(segment_matrices), len(segment_matrices))))
for i in range(len(segment_matrices)):
    for j in range(len(segment_matrices)):
        matrix_i = segment_matrices[i]
        matrix_j = segment_matrices[j]
        dist = np.sum(np.abs(matrix_i-matrix_j))
        consistency_matrix[i,j] = dist

# List of names for the matrices (replace with your actual list of names)
matrix_names = ["Age", "Need/Choice", "Awareness", "Reason/why", "Activity", "Speed/requirements",
                "Budget", "Consistency", "Tenure", "Switching propensity"]

# Initialize the distance matrix
consistency_matrix = np.zeros((len(segment_matrices), len(segment_matrices)))

for i in range(len(segment_matrices)):
    for j in range(len(segment_matrices)):
        matrix_i = segment_matrices[i]
        matrix_j = segment_matrices[j]
        dist = np.sum(np.abs(matrix_i - matrix_j))
        consistency_matrix[i, j] = dist
# Plot the consistency matrix
plt.figure(figsize=(8, 8))
plt.matshow(consistency_matrix, fignum=1)
plt.title("Consistencies")
plt.colorbar()
# Adding axis labels with custom tick labels
plt.xlabel('Question theme')
plt.ylabel('Question theme')
# Setting custom tick labels using the names from the list
plt.xticks(ticks=np.arange(len(matrix_names)), labels=matrix_names, rotation=90)
plt.yticks(ticks=np.arange(len(matrix_names)), labels=matrix_names)
plt.savefig("Consistency_matrix")
plt.show()

if make_plots:
    dendrogram_plot_test(consistency_matrix, "_consistency", "survey", matrix_names)

# Compute the summed affinity matrix
summed_affinity_matrix = sum_affinity_matrices(affinity_dict)
summed_select_affinity_matrix = sum_affinity_matrices(select_affinity_dict)

# Display the result
print(summed_affinity_matrix)

# if make_plots:
# Plot of affinity matrix
plt.matshow(summed_affinity_matrix)
plt.colorbar()
plt.show()

# Dendrogram
dendrogram_plot_test(summed_affinity_matrix, "mixed_", "survey", np.linspace(1,data_length,data_length))
# Dendrogram
dendrogram_plot_test(summed_select_affinity_matrix, "mixed_", "survey", np.linspace(1,data_length,data_length))



# def plot_pca_kmeans(affinity_matrix, n_components=2, n_clusters=4):
#     # Standardize the affinity matrix
#     scaler = StandardScaler()
#     affinity_matrix_scaled = scaler.fit_transform(affinity_matrix)
#
#     # Perform PCA to reduce to 2 components
#     pca = PCA(n_components=n_components)
#     principal_components = pca.fit_transform(affinity_matrix_scaled)
#
#     # Perform K-means clustering
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#     clusters = kmeans.fit_predict(principal_components)
#
#     # Plot the clusters on the first two principal components
#     plt.figure(figsize=(10, 8))
#     plt.scatter(principal_components[:, 0], principal_components[:, 1], c=clusters, cmap='viridis', s=50)
#     plt.title('Clusters on Principal Components')
#     plt.xlabel('Principal Component 1')
#     plt.ylabel('Principal Component 2')
#     plt.colorbar(label='Cluster')
#     plt.grid(True)
#     plt.show()
#
#     # Plot the principal component loadings
#     loadings = pca.components_.T
#     num_features = affinity_matrix.shape[1]
#     x_labels = [f"Feature {i + 1}" for i in range(num_features)]
#
#     plt.figure(figsize=(10, 6))
#     plt.bar(x_labels, loadings[:, 0], alpha=0.5, align='center', label='PC1')
#     plt.bar(x_labels, loadings[:, 1], alpha=0.5, align='center', label='PC2')
#     plt.title('Principal Component Loadings')
#     plt.xlabel('Features')
#     plt.ylabel('Loading Value')
#     plt.xticks(rotation=45)
#     plt.legend()
#     plt.show()
#
# # Example usage
# plot_pca_kmeans(summed_affinity_matrix)
# plot_pca_kmeans(summed_select_affinity_matrix)

