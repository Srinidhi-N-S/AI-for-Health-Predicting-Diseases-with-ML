# AI for Health: Predicting Diseases with ML

## Overview
"AI for Health: Predicting Diseases with ML" is an interactive machine learning application developed using **Streamlit**. It leverages machine learning models to predict the likelihood of diseases based on various health metrics. The project includes the following features:

- **Disease Prediction**: Allows users to input health data (such as age, glucose level, blood pressure, BMI) and predicts whether a person is at risk for a disease using models like **Naive Bayes** and **SVM (Support Vector Machine)**.
- **Cluster Analysis**: Enables users to upload a dataset for clustering analysis, grouping patients with similar health profiles based on features such as age, BMI, blood pressure, and more. The KMeans clustering algorithm is used to identify patterns and visualize clusters.

## Key Features
- **Disease Prediction**:
  - Choose between **Naive Bayes** and **SVM** models for disease prediction.
  - Input personal health metrics to assess disease risk.
  - Displays a result indicating whether the individual is **positive** or **negative** for disease.
  
- **Cluster Analysis**:
  - Upload custom datasets for clustering analysis.
  - Preprocess and scale the data before applying **KMeans clustering**.
  - Visualize the results using interactive scatter plots.
  - Option to save the clustered data for further analysis.
  
- **Machine Learning Models**:
  - Pre-trained models for **Naive Bayes** and **SVM** for disease prediction.
  - KMeans algorithm for grouping similar health profiles.

## Technologies Used
- **Python**: The primary programming language.
- **Streamlit**: A fast web framework for building machine learning web apps.
- **scikit-learn**: Used for machine learning models (Naive Bayes, SVM, KMeans).
- **Pandas**: For data manipulation and preprocessing.
- **Matplotlib** and **Seaborn**: For data visualization.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ai-for-health.git
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the Streamlit app:
   ```bash
   streamlit run app.py

## Usage
# Once the app is running:
- Visit the Home page for an overview of the application.
- On the Predict Disease page, input health metrics such as age, glucose level, blood pressure, and BMI to predict the likelihood of disease using the selected model.
- On the Cluster Analysis page, upload a CSV dataset with health metrics, select the number of clusters, and visualize the resulting clusters.
- The About page provides details about the project and the models used.

## Contributing
Contributions are welcome! Feel free to fork this repository, submit issues, or open pull requests.

## MIT License
This project is licensed under the MIT License.


You can now copy the entire block of text above into your **README** file. Remember to replace `"Srinidhi-070"` with your actual GitHub username. This will ensure that users can easily follow the instructions to clone, install dependencies, and run your app.
