#A. Load the Dataset

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns

# Load the dataset
data = pd.read_csv("wine_data.csv")

# Display first few rows
print(data.head())

# B.Preprocess the Data
# Check for missing values
print("Missing values:\n", data.isnull().sum())

# Impute missing values with the mean
data['pH'] = data['pH'].fillna(data['pH'].mean())
data['alcohol'] = data['alcohol'].fillna(data['alcohol'].mean())
data['sulfur_dioxide'] = data['sulfur_dioxide'].fillna(data['sulfur_dioxide'].mean())

# Select features for clustering
features = ['pH', 'alcohol', 'sulfur_dioxide']

# Scale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[features])

# Create a new DataFrame with scaled features
scaled_data = pd.DataFrame(scaled_features, columns=features)

print("Scaled Data:\n", scaled_data.head())

# C. Implement Clustering Algorithm
# Determine the optimal number of clusters using the Elbow Method
inertia = []
cluster_range = range(1, 11)

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(8, 5))
plt.plot(cluster_range, inertia, marker='o')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia (Sum of Squared Distances)')
plt.show()

# From the graph, choose the optimal number of clusters (e.g., 3)
optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
kmeans.fit(scaled_data)

# Add cluster labels to the original dataset
data['Cluster'] = kmeans.labels_

print("Cluster Labels:\n", data['Cluster'].value_counts())

# D.  Visualize the Clusters
# Scatter plot for visualizing clusters
plt.figure(figsize=(8, 5))
sns.scatterplot(
    x=data['pH'], y=data['alcohol'], 
    hue=data['Cluster'], palette='viridis'
)
plt.title('Clusters of Wines')
plt.xlabel('pH')
plt.ylabel('Alcohol Content')
plt.legend(title='Cluster')
plt.show()

# Pair plot for deeper visualization
sns.pairplot(data, vars=features, hue='Cluster', palette='viridis')
plt.show()

# E. Explanation of Results
# Calculate the Silhouette Score
sil_score = silhouette_score(scaled_data, data['Cluster'])
print(f"Silhouette Score: {sil_score}")
