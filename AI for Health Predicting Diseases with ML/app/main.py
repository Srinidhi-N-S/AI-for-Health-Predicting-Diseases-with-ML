import streamlit as st
from utils import load_model, preprocess_data
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

# App Configuration
st.set_page_config(page_title="AI for Health", layout="wide")

# Sidebar Navigation
menu = ["Home", "Predict Disease", "Cluster Analysis", "About"]
choice = st.sidebar.selectbox("Menu", menu)

# Home Page
if choice == "Home":
    st.title("AI for Health: Predicting Diseases with ML")
    st.markdown("""
        This application uses machine learning to predict diseases and analyze patient data.
        - **Prediction Models**: Naive Bayes, SVM
        - **Cluster Analysis**: Grouping similar patient profiles.
    """)

# Predict Disease Page
elif choice == "Predict Disease":
    st.title("Predict Disease")
    st.write("Provide patient data to predict the likelihood of disease.")
    
    # Input Form
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    glucose = st.number_input("Glucose Level", min_value=0.0)
    bp = st.number_input("Blood Pressure", min_value=0.0)
    bmi = st.number_input("BMI", min_value=0.0)
    
    model_choice = st.selectbox("Choose a Model", ["Naive Bayes", "SVM"])
    
    if st.button("Predict"):
        input_data = preprocess_data(pd.DataFrame([[age, glucose, bp, bmi]], 
                                                  columns=["Age", "Glucose", "BP", "BMI"]))
        model = load_model(f"models/{model_choice.lower()}_model.pkl")
        prediction = model.predict(input_data)[0]
        st.success(f"Prediction: {'Positive for Disease' if prediction else 'Negative for Disease'}")

# Cluster Analysis Page
elif choice == "Cluster Analysis":
    st.title("Cluster Analysis")
    st.write("Analyze and visualize patient clusters based on health metrics.")

    # File uploader for the clustering dataset
    uploaded_file = st.file_uploader("Upload your clustering dataset", type="csv")
    
    if uploaded_file is not None:
        # Load and display the uploaded dataset
        cluster_data = pd.read_csv(uploaded_file)
        st.write("### Dataset Preview")
        st.dataframe(cluster_data.head())
        
        # Preprocess Data
        st.write("### Data Preprocessing")
        # Make sure the required columns exist in the uploaded dataset
        required_columns = ["Age", "BMI", "Systolic_blood_pressure", "Weight"]
        if all(col in cluster_data.columns for col in required_columns):
            features = cluster_data[required_columns]
            features = features.dropna()  # Remove missing values
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)

            # KMeans Clustering
            st.write("### KMeans Clustering")
            n_clusters = st.slider("Select the number of clusters", min_value=2, max_value=10, value=3, step=1)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(scaled_features)

            # Add Cluster Labels to the Dataset
            cluster_data["Cluster"] = clusters
            st.write("### Clustered Data")
            st.dataframe(cluster_data)

            # Visualize Clusters
            st.write("### Cluster Visualization")
            plt.figure(figsize=(10, 6))
            sns.scatterplot(
                x=scaled_features[:, 0],
                y=scaled_features[:, 1],
                hue=clusters,
                palette="viridis",
                legend="full"
            )
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
            plt.title("Patient Clusters")
            st.pyplot(plt)

            # Option to Save Clustered Data
            if st.checkbox("Save Clustered Data"):
                cluster_data.to_csv("data/cluster_dataset.csv", index=False)
                st.success("Clustered data saved as 'clustered_data.csv'.")
        else:
            st.error("The dataset must contain the following columns: Age, BMI, Systolic_blood_pressure, and Weight.")
    else:
        st.error("Please upload the clustering dataset.")

# About Page
elif choice == "About":
    st.title("About")
    st.markdown("""
        This project demonstrates the power of machine learning in healthcare.
        - Developed using Python and Streamlit.
        - Models: Naive Bayes, SVM, and K-Means.
    """)
