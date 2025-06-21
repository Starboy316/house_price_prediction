import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os

# Create model folder if it doesn't exist
os.makedirs("model", exist_ok=True)

# Load dataset
print("🔹 Loading California Housing dataset...")
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

# Train and evaluate each model
for name, model in models.items():
    print(f"\n🚀 Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5

    print(f"📊 {name} Performance:")
    print(f"   MAE : {mae:.4f}")
    print(f"   MSE : {mse:.4f}")
    print(f"   RMSE: {rmse:.4f}")

    # Save model
    file_name = name.lower().replace(" ", "_").replace("xgboost", "xgb") + "_model.pkl"
    path = os.path.join("model", file_name)
    joblib.dump(model, path)
    print(f"✅ Saved to {path}")
