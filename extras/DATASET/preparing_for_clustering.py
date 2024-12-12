from sklearn.preprocessing import StandardScaler

# Select relevant features (update column names as per your dataset)
features = data[['Age', 'Weight', 'BMI', 'Systolic_blood_pressure']]  # Replace with actual column names

# Drop rows with missing values
features = features.dropna()

# Standardize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
