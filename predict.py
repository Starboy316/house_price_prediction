import joblib

def get_user_input():
    print("Enter the following details to predict house price:")

    MedInc = float(input("1. Median Income (e.g. 8.3): "))
    HouseAge = float(input("2. House Age (e.g. 41): "))
    AveRooms = float(input("3. Average Rooms (e.g. 6.0): "))
    AveBedrms = float(input("4. Average Bedrooms (e.g. 1.0): "))
    Population = float(input("5. Population (e.g. 980): "))
    AveOccup = float(input("6. Average Occupants per household (e.g. 3.2): "))
    Latitude = float(input("7. Latitude (e.g. 34.2): "))
    Longitude = float(input("8. Longitude (e.g. -118.4): "))

    return [[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]]

# Load model
model = joblib.load("model/price_model.pkl")

# Get input
features = get_user_input()

# Predict
prediction = model.predict(features)
print(f"\n💰 Predicted Median House Value: ${prediction[0]*100000:.2f}")
