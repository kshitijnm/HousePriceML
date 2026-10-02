import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# STEP 1: LOAD DATASET
# ============================================================

df = pd.read_csv("dataset/Housing.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ============================================================
# STEP 2: SEPARATE FEATURES AND TARGET
# ============================================================

# X = input features
# y = target variable (house price)

X = df.drop("price", axis=1)
y = df["price"]


# ============================================================
# STEP 3: ENCODE CATEGORICAL VARIABLES
# ============================================================

# Convert text categories into numerical columns

X = pd.get_dummies(
    X,
    drop_first=True,
    dtype=int
)

print("\nProcessed Features:")
print(X.head())

print("\nProcessed Columns:")
print(X.columns.tolist())

print("\nProcessed X shape:", X.shape)
print("Target y shape:", y.shape)


# ============================================================
# STEP 4: TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ============================================================
# STEP 5: CREATE LINEAR REGRESSION MODEL
# ============================================================

model = LinearRegression()


# ============================================================
# STEP 6: TRAIN THE MODEL
# ============================================================

model.fit(X_train, y_train)

print("\nModel training completed successfully!")


# ============================================================
# STEP 7: DISPLAY INTERCEPT
# ============================================================

print("\nIntercept:")
print(model.intercept_)


# ============================================================
# STEP 8: DISPLAY COEFFICIENTS
# ============================================================

print("\nCoefficients:")

for feature, coefficient in zip(X.columns, model.coef_):
    print(feature, ":", coefficient)


# ============================================================
# STEP 9: MAKE PREDICTIONS ON TEST DATA
# ============================================================

y_pred = model.predict(X_test)

print("\nPredicted Prices:")
print(y_pred[:10])

print("\nActual Prices:")
print(y_test.values[:10])


# ============================================================
# STEP 10: MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)


print("\nModel Evaluation:")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R²  :", r2)


# ============================================================
# STEP 11: TRAINING PERFORMANCE
# ============================================================

y_train_pred = model.predict(X_train)

train_mae = mean_absolute_error(y_train, y_train_pred)

train_rmse = np.sqrt(
    mean_squared_error(y_train, y_train_pred)
)

train_r2 = r2_score(y_train, y_train_pred)


print("\nTraining Performance:")
print("Train MAE :", train_mae)
print("Train RMSE:", train_rmse)
print("Train R²  :", train_r2)


# ============================================================
# STEP 12: TESTING PERFORMANCE
# ============================================================

print("\nTesting Performance:")
print("Test MAE  :", mae)
print("Test RMSE :", rmse)
print("Test R²   :", r2)


# ============================================================
# STEP 13: ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

# Perfect prediction line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")

plt.savefig(
    "graphs/actual_vs_predicted.png"
)

plt.show()


# ============================================================
# STEP 14: RESIDUAL PLOT
# ============================================================

residuals = y_test - y_pred

plt.figure(figsize=(8, 6))

plt.scatter(y_pred, residuals)

# Zero-error reference line
plt.axhline(y=0)

plt.xlabel("Predicted Price")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residual Plot")

plt.savefig(
    "graphs/residual_plot.png"
)

plt.show()


# ============================================================
# STEP 15: SAVE TRAINED MODEL
# ============================================================

joblib.dump(
    model,
    "model/house_price_model.pkl"
)

print("\nModel saved successfully!")

print("Saved file:")
print("model/house_price_model.pkl")


# ============================================================
# STEP 16: SAVE FEATURE NAMES
# ============================================================

feature_names = X.columns.tolist()

joblib.dump(
    feature_names,
    "model/feature_names.pkl"
)

print("\nFeature names saved successfully!")

print("Saved file:")
print("model/feature_names.pkl")


# ============================================================
# END
# ============================================================

print("\n========================================")
print("HOUSE PRICE ML PROJECT COMPLETED")
print("========================================")