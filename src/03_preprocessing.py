import pandas as pd

# Load the dataset
df = pd.read_csv("dataset/Housing.csv")

# Separate features and target
X = df.drop("price", axis=1)
y = df["price"]

# Convert categorical columns into numerical columns
X = pd.get_dummies(X, drop_first=True, dtype=int)

# Display the processed features
print("Processed Features (X):")
print(X.head())

# Display column names
print("\nProcessed columns:")
print(X.columns.tolist())

# Display shapes
print("\nX shape:", X.shape)
print("y shape:", y.shape)

from sklearn.model_selection import train_test_split

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Display the shapes
print("\nTraining data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)