# Airline Passenger Satisfaction Predictor (Streamlit Data Product)
# ------------------------------------------------------------------
# VERY SIMPLE app: user enters passenger info -> clicks Predict ->
# app shows SATISFIED or DISSATISFIED.
# It loads models/final_model.pkl saved by ml_model.py, so the SAME
# preprocessing (filling + one-hot columns) is applied here.
#
# Run with:  streamlit run app.py

import os
import joblib
import pandas as pd
import streamlit as st

# Load the saved bundle (model + columns + fill values + metrics)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
bundle = joblib.load(os.path.join(BASE_DIR, "models", "final_model.pkl"))

model = bundle["model"]              # the trained final model
model_columns = bundle["columns"]    # one-hot columns used in training
metrics = bundle["metrics"]          # accuracy/precision/recall/f1
comparison = bundle["comparison"]    # all 3 models
model_name = bundle["model_name"]    # e.g. "Random Forest"

st.title("Airline Passenger Satisfaction Predictor")
st.write("Enter passenger details and click **Predict Satisfaction**.")

# -------------------- User inputs (only real dataset columns) --------------------
age = st.number_input("Age", min_value=0, max_value=100, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
customer_type = st.selectbox("Customer Type",
                             ["Loyal Customer", "disloyal Customer"])
travel_type = st.selectbox("Type of Travel",
                           ["Business travel", "Personal Travel"])
travel_class = st.selectbox("Class", ["Business", "Eco", "Eco Plus"])
flight_distance = st.number_input("Flight Distance", min_value=0, value=500)
departure_delay = st.number_input("Departure Delay (minutes)",
                                  min_value=0, value=0)
arrival_delay = st.number_input("Arrival Delay (minutes)",
                                min_value=0.0, value=0.0)
seat_comfort = st.slider("Seat Comfort (0-5)", 0, 5, 3)
food_drink = st.slider("Food and Drink (0-5)", 0, 5, 3)
online_boarding = st.slider("Online Boarding (0-5)", 0, 5, 3)
cleanliness = st.slider("Cleanliness (0-5)", 0, 5, 3)
inflight_service = st.slider("Inflight Service (0-5)", 0, 5, 3)

# -------------------- Prediction --------------------
if st.button("Predict Satisfaction"):
    # Build one row with the EXACT original column names
    row = pd.DataFrame([{
        "Gender": gender,
        "Customer Type": customer_type,
        "Age": age,
        "Type of Travel": travel_type,
        "Class": travel_class,
        "Flight Distance": flight_distance,
        "Inflight wifi service": 3,               # fixed neutral default
        "Departure/Arrival time convenient": 3,   # (keeps the app short;
        "Ease of Online booking": 3,              #  user enters the key fields)
        "Gate location": 3,
        "Food and drink": food_drink,
        "Online boarding": online_boarding,
        "Seat comfort": seat_comfort,
        "Inflight entertainment": 3,
        "On-board service": 3,
        "Leg room service": 3,
        "Baggage handling": 3,
        "Checkin service": 3,
        "Inflight service": inflight_service,
        "Cleanliness": cleanliness,
        "Departure Delay in Minutes": departure_delay,
        "Arrival Delay in Minutes": arrival_delay,
    }])

    # SAME preprocessing as training: one-hot encode, align columns
    row = pd.get_dummies(row, drop_first=False)
    row = row.reindex(columns=model_columns, fill_value=0)

    pred = model.predict(row)[0]  # 1 = satisfied, 0 = dissatisfied
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(row)[0].max() * 100
    else:
        proba = None

    if pred == 1:
        st.success("Prediction: SATISFIED")
    else:
        st.error("Prediction: DISSATISFIED")
    if proba is not None:
        st.write(f"Prediction Probability: {proba:.1f}%")

# -------------------- Model Performance --------------------
st.header("Model Performance")
st.write(f"Final model: **{model_name}**")
st.write(f"Accuracy: {metrics['accuracy']:.4f}")
st.write(f"Precision: {metrics['precision']:.4f}")
st.write(f"Recall: {metrics['recall']:.4f}")
st.write(f"F1 Score: {metrics['f1']:.4f}")

st.subheader("Model Comparison")
comp_df = pd.DataFrame(comparison).T  # rows = models, cols = metrics
comp_df = comp_df[["accuracy", "precision", "recall", "f1"]]
st.table(comp_df)
