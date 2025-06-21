# 🏠 House Price Prediction - ML Project with Streamlit

## 📌 Project Overview

This is a beginner-level Machine Learning project that predicts California house prices using a Linear Regression model. The project includes:

- Data loading and preprocessing
- Model training and evaluation using Scikit-learn
- A CLI-based prediction tool
- An interactive web app using Streamlit

---

## 📁 Project Structure

```
house-price-prediction/
│
├── data/                  # (Optional) Local dataset storage
├── model/
│   └── price_model.pkl    # Saved ML model
├── main.py                # ML training and evaluation script
├── predict.py             # CLI tool for manual predictions
├── app.py                 # Streamlit web app
└── requirements.txt       # Python dependencies
```

---

## 🧠 ML Logic & Workflow

### 1. Dataset

We use the **California Housing dataset** from Scikit-learn, which includes features such as:

- Median Income
- House Age
- Average Rooms
- Average Bedrooms
- Population
- Average Occupants
- Latitude, Longitude

Target variable:

- **MedHouseVal**: Median house value (in 100,000 USD)

### 2. Model Training (main.py)

- Load dataset with `fetch_california_housing()`
- Perform train-test split (80/20)
- Train a `LinearRegression()` model
- Evaluate with MAE, MSE, and RMSE
- Save the trained model using `joblib`

### 3. CLI Prediction (predict.py)

- Loads the trained model from `model/price_model.pkl`
- Asks user to input 8 feature values
- Predicts and displays the price

### 4. Streamlit App (app.py)

- Uses sliders and number inputs to collect user data
- Predicts house price using the trained model
- Displays the prediction result in real time

---

## ▶️ How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python main.py
```

### 3. Run CLI prediction tool

```bash
python predict.py
```

### 4. Launch Streamlit web app

```bash
streamlit run app.py
```

---

## 📦 requirements.txt

```
pandas
scikit-learn
matplotlib
joblib
streamlit
```

---

## 🌟 Output Example

### CLI:

```
Enter the following details to predict house price:
1. Median Income (e.g. 8.3): 8.3
...
💰 Predicted Median House Value: $355,000.00
```

### Streamlit:

Interactive sliders → Click 'Predict Price' → Get house value in dollars

---

## ✅ Key Learning Outcomes

- Hands-on with Scikit-learn's real-world datasets
- Built and saved a regression model
- Developed both CLI and web app interfaces
- Used Streamlit for live ML prediction UI

---

