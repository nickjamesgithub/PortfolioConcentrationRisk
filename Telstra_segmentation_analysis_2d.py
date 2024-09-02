import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from Utilities import dendrogram_plot_test
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Read in data
data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Telstra\Data_clusters_engineered.csv")

# Step 1: Define a function to calculate the L1 (Manhattan) distance between two vectors
def l1_distance_matrix_q14(df):
    df_np = df.to_numpy()  # Convert the DataFrame to a NumPy array
    return np.sum(np.abs(df_np[:, np.newaxis] - df_np), axis=2)

def cosine_similarity_matrix(data):
    # Convert DataFrame to NumPy array
    data_np = data.to_numpy()
    norms = np.linalg.norm(data_np, axis=1)
    return np.nan_to_num(np.dot(data_np, data_np.T) / np.outer(norms, norms))

feature_list = ["Q1_encode", "Q4_encode", "Q8_encode",
                "Q14_encode", "Q15_encode", "Q16_encode", "Q17_encode", "Q23_encode", "Q30_encode",
                "Q31_encode", "Q35_encode", "Q41_encode", "Q42_encode"]

# Distance determination
l1_list = ["Q1_encode", "Q4_encode", "Q8_encode", "Q14_encode", "Q16_encode", "Q17_encode", "Q23_encode", "Q30_encode",
           "Q31_encode", "Q35_encode", "Q41_encode", "Q42_encode"]
cosine_list = ["Q15_encode"]

# Choose axes to condition upon
axis_1 = sorted(data["KPC_cluster"].unique())
axis_2 = sorted(data["Switching_cluster"].unique())

# Calculate the total number of combinations
num_combinations = len(axis_1) * len(axis_2)

# Initialize the 3D matrix to store affinity matrices for each feature
all_affinity_matrices = np.zeros((len(feature_list), num_combinations, num_combinations))

# Generate all possible combinations of (i, j, k)
combinations = [(i, j) for i in range(len(axis_1)) for j in range(len(axis_2))]

# Iterate over each feature and compute distances
for f, feature in enumerate(feature_list):
    print("Feature ", feature)
    distance_matrix = np.zeros((num_combinations, num_combinations))  # Initialize distance matrix for this feature

    for idx1, (i1, j1) in enumerate(combinations):
        for idx2, (i2, j2) in enumerate(combinations):
            # Choose first and second slice based on the two different triples
            slice_1 = data.loc[
                (data["KPC_cluster"] == axis_1[i1]) &
                (data["Switching_cluster"] == axis_2[j1]),
                data.columns[data.columns.str.startswith(feature)]
            ].values

            slice_2 = data.loc[
                (data["KPC_cluster"] == axis_1[i2]) &
                (data["Switching_cluster"] == axis_2[j2]),
                data.columns[data.columns.str.startswith(feature)]
            ].values

            # Compute means of the slices
            slice_1_mean = np.mean(slice_1, axis=0)
            slice_2_mean = np.mean(slice_2, axis=0)

            # Compute the distance
            if feature in l1_list:
                distance = np.sum(np.abs(slice_1_mean - slice_2_mean))
            else:
                norm1 = np.sum(np.abs(slice_1_mean))
                norm2 = np.sum(np.abs(slice_2_mean))
                if norm1 == 0 or norm2 == 0:
                    distance = 0
                else:
                    distance = np.dot(slice_1_mean, slice_2_mean) / (norm1 * norm2)


            # Store the distance in the distance matrix
            distance_matrix[idx1, idx2] = distance
            print(distance)

    # Convert the distance matrix to the affinity matrix
    max_distance = np.max(distance_matrix)
    if max_distance > 0:  # Avoid division by zero
        affinity_matrix = 1 - (distance_matrix / max_distance)
    else:
        affinity_matrix = np.ones_like(distance_matrix)  # If all distances are zero, affinity is 1

    # Plot affinity matrix
    plt.matshow(affinity_matrix)
    plt.title("Grid_distance_"+feature_list[f])
    plt.savefig("Grid_distance_" + feature_list[f])
    plt.close()

    # Store the affinity matrix
    all_affinity_matrices[f, :, :] = affinity_matrix

# Form dense affinity matrix
dense_affinity_matrix = np.sum(all_affinity_matrices, axis=0)

# Generate all possible combinations of (i, j, k)
axis_1 = sorted(data["KPC_cluster"].unique())
axis_2 = sorted(data["Switching_cluster"].unique())
combinations = [(i, j) for i in range(len(axis_1)) for j in range(len(axis_2))]

# Plotting the summed affinity matrix
plt.figure(figsize=(10, 8))
plt.imshow(dense_affinity_matrix, cmap='viridis', interpolation='none')
plt.colorbar(label='Summed Affinity')

# Annotating the plot with the segment indices
for idx, (i, j) in enumerate(combinations):
    plt.text(idx, idx, f'({i},{j})', ha='center', va='center', color='white', fontsize=8)

plt.title('Summed Affinity Matrix with Segment Indices')
plt.xlabel('Segment Index')
plt.ylabel('Segment Index')
plt.savefig("Dense_affinity_matrix")
plt.show()

# Dendrogram segmentation with new combination labels
dendrogram_plot_test(dense_affinity_matrix, "_Conditional_segmentation_numbered_", "Survey", combinations)

# Define the labels for each axis
axis_1_labels = ["Price", "Performance", "Convenience", "Offering"]  # Replace with actual labels for axis_1
axis_2_labels = ["Switch_0-2", "Switch_8+", "Switch_2-7"]  # Replace with actual labels for axis_2

# Check that the number of labels matches the number of unique elements in each axis
if len(axis_1_labels) != len(axis_1):
    raise ValueError("The length of axis_1_labels must match the number of unique elements in axis_1.")
if len(axis_2_labels) != len(axis_2):
    raise ValueError("The length of axis_2_labels must match the number of unique elements in axis_2.")

# Generate new combinations using the labels
new_combinations = [(axis_1_labels[i], axis_2_labels[j]) for i in range(len(axis_1)) for j in range(len(axis_2))]

# Example: Replace the old combination tuples in the plot annotation with new labels
plt.figure(figsize=(10, 8))
plt.imshow(dense_affinity_matrix, cmap='viridis', interpolation='none')
plt.colorbar(label='Summed Affinity')

# Annotating the plot with the new combination labels
for idx, (label1, label2) in enumerate(new_combinations):
    plt.text(idx, idx, f'({label1},{label2})', ha='center', va='center', color='white', fontsize=8)

plt.title('Summed Affinity Matrix with New Label Combinations')
plt.xlabel('Segment Index')
plt.ylabel('Segment Index')
plt.savefig("Dense_affinity_matrix_with_new_label_combinations")
plt.show()

# Dendrogram segmentation with new combination labels
dendrogram_plot_test(dense_affinity_matrix, "_Conditional_segmentation_theme_2d", "Survey", new_combinations)

# K means
kmeans = KMeans(n_clusters=5, random_state=42).fit(dense_affinity_matrix)
labels_affinity = kmeans.labels_

# Map each combination to its corresponding cluster label
combination_to_cluster = {comb: label for comb, label in zip(combinations, labels_affinity)}
# Initialize a new column in the original data to store the cluster labels
data['Cluster_Label'] = np.nan

# Iterate through each row in the original data and assign the corresponding cluster label
for i, row in data.iterrows():
    # Get the values for axis_1, axis_2, and axis_3 for this row
    comb = (row['KPC_cluster'], row['Switching_cluster'])
    # Assign the corresponding cluster label based on the combination
    if comb in combination_to_cluster:
        data.at[i, 'Cluster_Label'] = combination_to_cluster[comb]

# Convert 'Cluster_Label' column to integer type (optional)
data['Cluster_Label'] = data['Cluster_Label'].astype(int)
# Now, the 'Cluster_Label' column in the data DataFrame contains the appropriate cluster label for each customer
# Save the updated DataFrame (optional)
data.to_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Telstra\Data_with_cluster_projections_2d.csv", index=False)

x=1
y=2
