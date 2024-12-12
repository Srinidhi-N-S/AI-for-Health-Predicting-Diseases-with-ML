import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def run_cluster_analysis():
    # Load the data
    try:
        data = pd.read_csv("data/cluster_dataset.csv")
        st.write("### Dataset Preview")
        st.dataframe(data.head())
    except FileNotFoundError:
        st.error("Dataset not found! Please upload the clustering dataset.")
        return

    # Preprocess the data
    st.write("### Data Preprocessing")
    features = data[['Age', 'BMI', 'Systolic_BP', 'Diastolic_BP']]
    features = features.dropna()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Perform clustering
    n_clusters = st.slider("Select the number of clusters", 2, 10, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    data['Cluster'] = clusters
    st.write("### Clustered Data")
    st.dataframe(data)

    # Visualize the clusters
    st.write("### Cluster Visualization")
    fig, ax = plt.subplots()
    sns.scatterplot(x=scaled_features[:, 0], y=scaled_features[:, 1], hue=clusters, palette="viridis", ax=ax)
    plt.title("Clusters Visualization")
    st.pyplot(fig)

    # Save clustered data
    if st.checkbox("Save Clustered Data"):
        data.to_csv("data/clustered_data.csv", index=False)
        st.success("Clustered data saved as 'clustered_data.csv'.")
