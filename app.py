import streamlit as st
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="House Price Predictor", layout="wide")

# Load models
MODEL_PATHS = {
    "Linear Regression": "model/price_model.pkl",
    "Random Forest": "model/rf_model.pkl",
    "XGBoost": "model/xgb_model.pkl"
}

st.title("🏠 California House Price Predictor")
st.markdown("Enter details below or upload a CSV to predict house prices.")

# Choose model
model_choice = st.selectbox("🔀 Choose a model", list(MODEL_PATHS.keys()))
model = joblib.load(MODEL_PATHS[model_choice])

# Input sliders for individual prediction
st.markdown("## 🧮 Predict Single House Price")
with st.form("predict_form"):
    col1, col2, col3, col4 = st.columns(4)

    MedInc = col1.slider("Median Income (10k)", 0.0, 15.0, 5.0)
    HouseAge = col2.slider("House Age", 1, 52, 25)
    AveRooms = col3.slider("Avg Rooms", 1.0, 10.0, 5.0)
    AveBedrms = col4.slider("Avg Bedrooms", 0.5, 5.0, 1.0)

    col5, col6, col7, col8 = st.columns(4)
    Population = col5.number_input("Population", 100, 40000, 1000)
    AveOccup = col6.slider("Avg Occupants", 1.0, 10.0, 3.0)
    Latitude = col7.slider("Latitude", 32.0, 42.0, 36.0)
    Longitude = col8.slider("Longitude", -124.0, -114.0, -120.0)

    submitted = st.form_submit_button("🔮 Predict")
    if submitted:
        input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                                Population, AveOccup, Latitude, Longitude]])
        prediction = model.predict(input_data)[0] * 100000
        st.success(f"💰 Estimated Median House Value: **${prediction:,.2f}**")

# CSV Batch Prediction
st.markdown("## 📁 Batch Prediction via CSV Upload")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    required_cols = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                     'Population', 'AveOccup', 'Latitude', 'Longitude']

    if all(col in df.columns for col in required_cols):
        predictions = model.predict(df[required_cols])
        df['PredictedPrice'] = predictions * 100000
        st.success("✅ Predictions complete!")
        st.dataframe(df)

        # Heatmap Visualization
        st.markdown("## 🌍 Heatmap of Predicted Prices by Location")
        fig, ax = plt.subplots(figsize=(10, 6))
        heat = sns.scatterplot(data=df, x="Longitude", y="Latitude",
                               hue="PredictedPrice", palette="coolwarm", size="PredictedPrice", sizes=(40, 200), ax=ax)
        plt.title("House Price Heatmap")
        plt.grid(True)
        st.pyplot(fig)

    else:
        st.error(f"❌ CSV must contain these columns: {', '.join(required_cols)}")
