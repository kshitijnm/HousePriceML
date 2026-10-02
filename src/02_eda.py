import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("dataset/Housing.csv")

# Scatter plot: Area vs Price
plt.scatter(df["area"], df["price"])

plt.xlabel("Area")
plt.ylabel("Price")
plt.title("House Area vs Price")

plt.show()


# Correlation matrix of numerical columns
correlation = df.corr(numeric_only=True)

print("\nCorrelation Matrix:")
print(correlation)

# Create heatmap
plt.figure(figsize=(10, 7))

sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")

plt.title("Correlation Heatmap")

plt.show()

# Average price by air conditioning
plt.figure(figsize=(7, 5))

sns.barplot(
    data=df,
    x="airconditioning",
    y="price"
)

plt.title("Average House Price by Air Conditioning")
plt.xlabel("Air Conditioning")
plt.ylabel("Average Price")

plt.show()

# Average price by furnishing status
plt.figure(figsize=(8, 5))

sns.barplot(
    data=df,
    x="furnishingstatus",
    y="price"
)

plt.title("Average House Price by Furnishing Status")
plt.xlabel("Furnishing Status")
plt.ylabel("Average Price")

plt.show()