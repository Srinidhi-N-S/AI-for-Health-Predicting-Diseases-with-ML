import pandas as pd

# Load the clustered dataset
file_path = "health_data.csv"  # Update with the correct path
data = pd.read_csv(file_path)

# Preview the dataset
print(data.head())

# Display column names
print(data.columns)
