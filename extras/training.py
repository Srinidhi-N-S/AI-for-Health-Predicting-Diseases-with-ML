import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("patient_dataset.csv")

# Separate features and target
X = data.drop(columns=["Disease"])
y = data["Disease"]

# Identify categorical features
categorical_features = X.select_dtypes(include=['object']).columns

# Create a ColumnTransformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X.select_dtypes(exclude=['object']).columns),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
    ])

# Apply preprocessing
X_processed = preprocessor.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, random_state=42)

# Visualizations

# Histogram for Age
plt.figure(figsize=(8, 6))
sns.histplot(data['Age'], kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Box plot for Age by Gender
plt.figure(figsize=(8, 6))
sns.boxplot(x='Gender', y='Age', data=data)
plt.title('Age Distribution by Gender')
plt.xlabel('Gender')
plt.ylabel('Age')
plt.show()

# Count plot for Disease
plt.figure(figsize=(8, 6))
sns.countplot(x='Disease', data=data)
plt.title('Disease Counts')
plt.xlabel('Disease')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.show()

plt.figure(figsize=(10, 8))
numerical_features = data.select_dtypes(include=['number']).columns  # Select numerical features
correlation_matrix = data[numerical_features].corr()  # Calculate correlation for numerical features
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
