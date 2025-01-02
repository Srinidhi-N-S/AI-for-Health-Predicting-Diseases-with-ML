import streamlit as st
from utils import load_model, preprocess_data
import pandas as pd
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import os

# App Configuration
st.set_page_config(page_title="AI for Health", layout="wide")

# Inject custom CSS for dropdown menu and general cursor changes
st.markdown(
    """
    <style>
    /* Change cursor to hand pointer for the dropdown menu */
    .stSelectbox [role='button'] {
        cursor: pointer;
    }
    /* Style for general hover effect */
    div:hover {
        cursor: pointer;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar Navigation
menu = ["Home", "Predict Disease", "Cluster Analysis", "About"]
choice = st.sidebar.selectbox("Menu", menu, key="unique_menu_key")

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

# Predict Disease Page
elif choice == "Predict Disease":
    st.title("Predict Disease")
    st.write("Provide patient data to predict the likelihood of disease.")

    # Input Form
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    weight = st.number_input("Weight (in kg)", min_value=0.0, step=0.1)
    bmi = st.number_input("BMI", min_value=0.0, step=0.1)
    systolic_bp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=0.0, step=0.1)

    model_choice = st.selectbox("Choose a Model", ["Naive Bayes", "SVM"])

    if st.button("Predict"):
        try:
            # Load Scaler and Preprocess Data
            scaler_path = "models/scaler.pkl"
            if not os.path.exists(scaler_path):
                st.error("Scaler file not found. Please ensure the scaler is saved at 'models/scaler.pkl'.")
            else:
                input_data = preprocess_data(
                    pd.DataFrame([[age, weight, bmi, systolic_bp]], columns=["Age", "Weight", "BMI", "Systolic_blood_pressure"]),
                    scaler_path
                )

                # Load Model
                model_path = f"models/{model_choice.lower()}_model.pkl"
                if not os.path.exists(model_path):
                    st.error(f"Model file for {model_choice} not found at {model_path}.")
                else:
                    model = load_model(model_path)
                    prediction = model.predict(input_data)[0]

                    # Visualize Prediction with Gauge Chart
                    st.write("### Prediction Gauge Chart")
                    if prediction == 1:
                        label = "Positive for Disease"
                        color = "red"
                    else:
                        label = "Negative for Disease"
                        color = "green"
                    
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prediction,
                        title={"text": "Disease Prediction", "font": {"size": 24}},
                        gauge={
                            "axis": {"range": [0, 1]},
                            "bar": {"color": color},
                            "steps": [
                                {"range": [0, 0.5], "color": "green"},
                                {"range": [0.5, 1], "color": "red"}
                            ]
                        }
                    ))
                    st.plotly_chart(fig)
                    
                    st.write(f"The model predicts: **{label}**")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# Cluster Analysis Page
elif choice == "Cluster Analysis":
    st.title("Cluster Analysis")
    st.write("Analyze and visualize patient clusters based on health metrics.")

    # File uploader for clustering dataset
    uploaded_file = st.file_uploader("Upload your clustering dataset", type="csv")

    if uploaded_file is not None:
        try:
            # Load and display the uploaded dataset
            cluster_data = pd.read_csv(uploaded_file)
            st.write("### Dataset Preview")
            st.dataframe(cluster_data.head())

            # Preprocess Data
            st.write("### Data Preprocessing")
            required_columns = ["Age", "BMI", "Systolic_blood_pressure", "Weight"]

            if all(col in cluster_data.columns for col in required_columns):
                features = cluster_data[required_columns].dropna()  # Remove missing values
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

        except Exception as e:
            st.error(f"An error occurred during clustering: {e}")
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
