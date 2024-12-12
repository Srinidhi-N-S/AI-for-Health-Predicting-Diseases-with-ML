import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Load dataset
data = pd.read_csv("healthcare_dataset.csv")

# Drop non-informative columns
columns_to_drop = ["Name", "Doctor", "Hospital", "Room Number", "Date of Admission", "Discharge Date"]
data = data.drop(columns=columns_to_drop)

# Encode non-numeric columns
non_numeric_columns = data.select_dtypes(include=['object']).columns
label_encoders = {}
for column in non_numeric_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le  # Save encoder for later use

# Separate features and target
X = data.drop(columns=["Medical Condition"])
y = data["Medical Condition"]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Evaluate Naive Bayes
nb_pred = nb_model.predict(X_test)
print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_pred))
print(classification_report(y_test, nb_pred))

# Train SVM model
svm_model = SVC(kernel="linear", random_state=42)
svm_model.fit(X_train, y_train)

# Evaluate SVM
svm_pred = svm_model.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))
print(classification_report(y_test, svm_pred))

# Create models directory
os.makedirs("models", exist_ok=True)

# Save the Naive Bayes model
joblib.dump(nb_model, "naive_bayes_model.pkl")

# Save the SVM model
joblib.dump(svm_model, "svm_model.pkl")

print("Models have been saved as .pkl files in the 'models' directory.")
