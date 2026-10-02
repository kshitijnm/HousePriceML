import pandas as pd

# Load the dataset
df = pd.read_csv("dataset/Housing.csv")

# Display the first 5 rows
print(df.head())

# Display number of rows and columns
print("Dataset shape:", df.shape)

# Display information about the dataset
df.info()

# Display statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check unique values in categorical columns
print("\nUnique values in categorical columns:")

categorical_columns = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
    "furnishingstatus"
]

for column in categorical_columns:
    print(column, ":", df[column].unique())