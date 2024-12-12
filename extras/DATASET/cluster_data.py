import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Load Dataset
data = pd.read_csv("health_data.csv")

# Step 2: Select Features for Clustering
# Choose relevant numeric features
features = data[['Age', 'BMI', 'Systolic_blood_pressure', 'Weight']]  # Modify as needed

# Step 3: Handle Missing Values
features = features.dropna()

# Step 4: Scale Features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Step 5: Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # Adjust clusters (e.g., 3 groups)
clusters = kmeans.fit_predict(scaled_features)

# Step 6: Add Cluster Labels to DataFrame
data['Cluster'] = clusters

# Step 7: Save to CSV
data.to_csv("cluster_data.csv", index=False)
print("Clustered data saved to 'cluster_data.csv'.")
