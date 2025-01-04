import streamlit as st
from utils import load_model, preprocess_data
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
import requests
from io import BytesIO
import joblib

# App Configuration
st.set_page_config(page_title="AI for Health", layout="wide")

# Inject custom CSS for dropdown menu and hover effects
st.markdown(
    """
    <style>
    .stSelectbox [role='button'] { cursor: pointer; }
    div:hover { cursor: pointer; }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar Navigation
menu = ["Home", "Predict Disease", "Cluster Analysis", "About"]
choice = st.sidebar.selectbox("Menu", menu)

# Helper function to ensure directory exists
def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Home Page
if choice == "Home":
    st.title("AI for Health: Predicting Diseases with ML")
    st.markdown("""
        This application leverages machine learning for disease prediction and data analysis:
        - **Prediction Models**: Naive Bayes, SVM.
        - **Cluster Analysis**: Grouping patient profiles based on metrics.
    """)

    # Add an image to the home page
    image_url = "https://media.istockphoto.com/id/1165479316/vector/medical-concept-idea-design.jpg?s=612x612&w=0&k=20&c=aFGhZbjoQMRge7BVTGunXrA7ro7O1AATytLKp6Jht-U="
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        st.image(image, caption='AI for Health', use_container_width=True)
    except Exception as e:
        st.error(f"An error occurred while loading the image: {e}")

# Predict Disease Page
elif choice == "Predict Disease":
    st.title("Predict Disease")
    st.write("Provide patient data to predict the likelihood of diseases.")

    # Input Form
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    weight = st.number_input("Weight (in kg)", min_value=0.0, step=0.1)
    bmi = st.number_input("BMI", min_value=0.0, step=0.1)
    systolic_bp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=0.0, step=0.1)

    if st.button("Predict"):
        try:
            # Preprocess input
            input_data = pd.DataFrame([[age, weight, bmi, systolic_bp]], 
                                      columns=["Age", "Weight", "BMI", "Systolic_blood_pressure"])
            scaler = joblib.load(r"C:\Users\nssri\OneDrive\Desktop\Projects\AI for Health Predicting Diseases with ML\app\models\scaler.pkl")
            model = joblib.load(r"C:\Users\nssri\OneDrive\Desktop\Projects\AI for Health Predicting Diseases with ML\app\models\multi_class_model.pkl")

            input_scaled = scaler.transform(input_data)

            # Predict disease
            prediction = model.predict(input_scaled)[0]
            st.success(f"Prediction: {prediction}")

            # Optional: Show prediction probabilities
            probabilities = model.predict_proba(input_scaled)[0]
            disease_labels = model.classes_
            prob_df = pd.DataFrame({"Disease": disease_labels, "Probability": probabilities})
            st.bar_chart(prob_df.set_index("Disease"))
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# Cluster Analysis Page
elif choice == "Cluster Analysis":
    st.title("Cluster Analysis")
    st.write("Analyze and visualize patient clusters based on health metrics.")

    uploaded_file = st.file_uploader("Upload your clustering dataset", type="csv")

    if uploaded_file is not None:
        try:
            cluster_data = pd.read_csv(uploaded_file)
            st.write("### Dataset Preview")
            st.dataframe(cluster_data.head())

            required_columns = ["Age", "BMI", "Systolic_blood_pressure", "Weight"]
            if all(col in cluster_data.columns for col in required_columns):
                features = cluster_data[required_columns].dropna()
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(features)

                # KMeans Clustering
                n_clusters = st.slider("Select the number of clusters", min_value=2, max_value=10, value=3, step=1)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(scaled_features)

                cluster_data["Cluster"] = clusters
                st.write("### Clustered Data")
                st.dataframe(cluster_data)

                # Visualize Clusters using PCA
                pca = PCA(n_components=2)
                pca_features = pca.fit_transform(scaled_features)
                pca_df = pd.DataFrame(pca_features, columns=["PCA1", "PCA2"])
                pca_df["Cluster"] = clusters

                fig = px.scatter(pca_df, x="PCA1", y="PCA2", color=pca_df["Cluster"].astype(str), 
                                 title="2D PCA of Clusters", labels={"Cluster": "Cluster Group"}, 
                                 color_discrete_sequence=px.colors.qualitative.Set1)
                st.plotly_chart(fig)

                # Display Silhouette Score
                silhouette_avg = silhouette_score(scaled_features, clusters)
                st.write(f"### Silhouette Score: {silhouette_avg:.2f}")

                # Feature Importance Heatmap
                feature_importance = pd.DataFrame(scaled_features, columns=required_columns)
                feature_importance["Cluster"] = clusters
                heatmap_data = feature_importance.groupby("Cluster").mean()

                plt.figure(figsize=(10, 6))
                sns.heatmap(heatmap_data, annot=True, cmap="coolwarm")
                plt.title("Cluster-wise Feature Importance")
                st.pyplot(plt)

        except Exception as e:
            st.error(f"An error occurred during clustering: {e}")
    else:
        st.error("Please upload the clustering dataset.")

# About Page
elif choice == "About":
    st.title("About")
    st.markdown("""
        This project demonstrates the power of machine learning in healthcare:
        - **Prediction Models**: Naive Bayes, SVM.
        - **Cluster Analysis**: K-Means for grouping patient data.
        - **Technologies Used**: Python, Streamlit.
    """)

    st.markdown("""
        ### Key Highlights:
        - **Customization**: The application supports flexible dataset inputs and real-time visualization.
        - **User-Friendly**: Designed with an intuitive interface using Streamlit.
        - **Expandability**: Future-ready for adding more models and clustering techniques.
        - **Impact**: Aims to aid medical professionals in decision-making with data-driven insights.
    """)

    st.markdown("""
        ### Future Enhancements:
        - Integration with electronic health records (EHR) systems.
        - Implementation of deep learning models for more accurate predictions.
        - Real-time data streaming for continuous health monitoring.
    """)

    # Add an image to the About page
    about_image_url = "https://media.istockphoto.com/id/1278978974/photo/abstract-medical-background.jpg?s=612x612&w=0&k=20&c=jH7OYHgIjD2_lFeXf7EGYWeYj0bi6PvPOoBP2GR78RY="
    try:
        response = requests.get(about_image_url)
        about_image = Image.open(BytesIO(response.content))
        st.image(about_image, caption='AI and Healthcare', use_container_width=True)
    except Exception as e:
        st.error(f"An error occurred while loading the image: {e}")
