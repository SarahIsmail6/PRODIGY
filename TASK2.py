import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
input_input_datasetset = pd.read_csv('Mall_Customers.csv')
num_columns = input_input_datasetset.select_dtypes(include=[np.number]).columns
non_numeric_columns = input_dataset.select_dtypes(exclude=[np.number]).columns
input_dataset[num_columns] = input_dataset[num_columns].fillna(input_dataset[num_columns].mean())
label_encoder = LabelEncoder()
for column in non_numeric_columns:
    input_dataset[column] = label_encoder.fit_transform(input_dataset[column].astype(str))
features = ['Age', 'Annual Income', 'Spending Score']
X = input_dataset[features]

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), sse, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('SSE')
    plt.title('Elbow Method')
    plt.show()
    silhouette_scores = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score Method')
plt.show()
plt.figure(figsize=(10, 6))
sns.scatterplot(input_dataset=input_dataset, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='viridis')
plt.title('Clusters of Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_centers_df = pd.input_datasetFrame(cluster_centers, columns=features)
cluster_centers_df['Cluster'] = range(optimal_k)
plt.figure(figsize=(10, 6))
sns.heatmap(cluster_centers_df.set_index('Cluster'), annot=True, cmap='coolwarm')
plt.title('Cluster Centers')
plt.show()
sns.pairplot(input_dataset, hue='Cluster', palette='viridis', vars=features)
plt.show()
plt.figure(figsize=(15, 10))
for i, feature in enumerate(features):
    plt.subplot(2, 2, i+1)
    sns.boxplot(x='Cluster', y=feature, input_dataset=input_dataset, palette='viridis')
    plt.title(f'Distribution of {feature} in Clusters')
plt.tight_layout()
plt.show()
details = input_dataset.groupby('Cluster')[features].mean().reset_index()
print(details)
input_dataset.to_csv('clustered_customers.csv', index=False)