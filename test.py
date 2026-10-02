import pandas as pd
import joblib


# ============================================================
# STEP 1: LOAD THE TRAINED MODEL
# ============================================================

model = joblib.load("model/house_price_model.pkl")
feature_names = joblib.load("model/feature_names.pkl")

print("\nMultiple Linear Regression model loaded successfully!")
print("Feature structure loaded successfully!")


# ============================================================
# STEP 2: INPUT VALIDATION FUNCTIONS
# ============================================================

def get_positive_float(message):
    while True:
        try:
            value = float(input(message))

            if value > 0:
                return value

            print("Please enter a value greater than 0.")

        except ValueError:
            print("Please enter a valid number.")


def get_non_negative_int(message):
    while True:
        try:
            value = int(input(message))

            if value >= 0:
                return value

            print("Please enter 0 or a positive number.")

        except ValueError:
            print("Please enter a valid whole number.")


def get_yes_no(message):
    while True:
        value = input(message).strip().lower()

        if value in ["yes", "no"]:
            return value

        print("Please enter only yes or no.")


def get_furnishing_status():
    while True:
        value = input(
            "Enter furnishing status "
            "(furnished/semi-furnished/unfurnished): "
        ).strip().lower()

        if value in ["furnished", "semi-furnished", "unfurnished"]:
            return value

        print(
            "Please enter furnished, semi-furnished, "
            "or unfurnished."
        )


# ============================================================
# STEP 3: ENTER NEW HOUSE DETAILS
# ============================================================

print("\n========================================")
print("ENTER HOUSE DETAILS")
print("========================================")

area = get_positive_float("Enter house area: ")

bedrooms = get_non_negative_int(
    "Enter number of bedrooms: "
)

bathrooms = get_non_negative_int(
    "Enter number of bathrooms: "
)

stories = get_non_negative_int(
    "Enter number of stories: "
)

parking = get_non_negative_int(
    "Enter number of parking spaces: "
)

mainroad = get_yes_no(
    "Is the house connected to main road? (yes/no): "
)

guestroom = get_yes_no(
    "Does the house have a guest room? (yes/no): "
)

basement = get_yes_no(
    "Does the house have a basement? (yes/no): "
)

hotwaterheating = get_yes_no(
    "Does the house have hot water heating? (yes/no): "
)

airconditioning = get_yes_no(
    "Does the house have air conditioning? (yes/no): "
)

prefarea = get_yes_no(
    "Is the house in a preferred area? (yes/no): "
)

furnishingstatus = get_furnishing_status()


# ============================================================
# STEP 4: CREATE INPUT DATAFRAME
# ============================================================

new_house = pd.DataFrame({
    "area": [area],
    "bedrooms": [bedrooms],
    "bathrooms": [bathrooms],
    "stories": [stories],
    "parking": [parking],
    "mainroad": [mainroad],
    "guestroom": [guestroom],
    "basement": [basement],
    "hotwaterheating": [hotwaterheating],
    "airconditioning": [airconditioning],
    "prefarea": [prefarea],
    "furnishingstatus": [furnishingstatus]
})


# ============================================================
# STEP 5: ENCODE CATEGORICAL VARIABLES
# ============================================================

new_house = pd.get_dummies(
    new_house,
    drop_first=True,
    dtype=int
)


# ============================================================
# STEP 6: MATCH MODEL'S FEATURE STRUCTURE
# ============================================================

new_house = new_house.reindex(
    columns=feature_names,
    fill_value=0
)


# ============================================================
# STEP 7: MAKE PREDICTION
# ============================================================

predicted_price = model.predict(new_house)


# ============================================================
# STEP 8: DISPLAY RESULT
# ============================================================

print("\n========================================")
print("HOUSE PRICE PREDICTION")
print("========================================")

print(
    "Predicted House Price: ₹",
    f"{predicted_price[0]:,.2f}"
)

print("========================================")
print("Prediction completed successfully!")