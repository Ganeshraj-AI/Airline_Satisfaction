# Airline Passenger Satisfaction Prediction

## 1. Problem Statement

Airlines collect passenger feedback after flights, but reading every survey
manually is slow. This project predicts whether a passenger is **satisfied**
or **dissatisfied** from their trip details (age, class, delays) and service
ratings (seat comfort, cleanliness, food, online boarding, ...), so the
airline can spot problems early and improve service.

## 2. Objective

Build a binary classification system: given one passenger's information,
predict `1 = Satisfied` or `0 = Dissatisfied`, and show the result in a
simple Streamlit web app.

## 3. Dataset

File: `data/airline_passenger_satisfaction.csv`
(public Kaggle "Airline Passenger Satisfaction" survey).

* Records: **129,880 rows, 25 columns**
* After cleaning: **129,880 rows, 23 columns** (2 ID columns dropped,
  0 duplicate rows found)
* Features used: **22** (4 categorical + 18 numerical)
* Target variable: `satisfaction` → `satisfied` (1) /
  `neutral or dissatisfied` (0)
* Class balance: Dissatisfied (0) = 73,452, Satisfied (1) = 56,428
  → satisfaction rate **43.45%**
* Categorical columns: Gender, Customer Type, Type of Travel, Class
* Numerical columns: Age, Flight Distance, 14 service ratings (0–5),
  Departure Delay in Minutes, Arrival Delay in Minutes
* Missing values: `Arrival Delay in Minutes` had **393** missing values
  (filled with median); all other columns had none.

## 4. Data Engineering

All steps live in `ml_model.py` (one file, beginner-friendly):

1. **Load + inspect** with pandas (`shape`, `head`, column names, data types,
   target value counts).
2. **Drop ID columns** (`Feature 1`, `id`) — unique per row, so they have no
   predictive value and would only let the model memorise rows.
3. **Missing values** — numerical → median, categorical → mode. Only
   `Arrival Delay in Minutes` had missing values (393, filled with median).
4. **Duplicates** — `df.duplicated().sum()` found 0, so nothing was removed.
5. **Feature selection** — `X` = all 22 input columns,
   `y` = `satisfaction` mapped to 1/0.
6. **Encoding** — `pd.get_dummies()` turns Gender, Customer Type,
   Type of Travel, Class into 0/1 columns (27 columns after encoding).
7. **Train/test split** — 80% train (103,904 rows) / 20% test (25,976 rows),
   `random_state=42`, `stratify=y` so both sets keep the same
   satisfied/dissatisfied ratio.
8. **Data leakage prevention** — the split happens BEFORE preprocessing;
   medians/modes are computed on the TRAIN set only and applied to the test
   set; the test set is only aligned to the train columns via `reindex`.

## 5. ML Models

| Model | Idea (one line) |
|---|---|
| Logistic Regression | Simple baseline classification model; learns a weighted sum passed through a logistic curve. |
| Decision Tree | Tree-based model; learns if/else rules like "if Online Boarding < 3 → dissatisfied". |
| Random Forest | Combination of 100 decision trees (`n_estimators=100`) that vote together. |

No hyperparameter tuning — default settings (plus `random_state=42`) to stay
explainable.

## 6. Model Evaluation (test set, 25,976 passengers)

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 0.8718 | 0.8673 | 0.8324 | 0.8495 |
| Decision Tree | 0.9463 | 0.9357 | 0.9412 | 0.9384 |
| Random Forest | 0.9627 | 0.9717 | 0.9414 | 0.9563 |

Confusion matrix of the final model (Random Forest),
rows = actual, columns = predicted:

|  | Predicted Dissatisfied | Predicted Satisfied |
|---|---|---|
| Actual Dissatisfied | 14,381 (TN) | 309 (FP) |
| Actual Satisfied | 661 (FN) | 10,625 (TP) |

(See `models/confusion_matrix.png` for the heatmap.)

Top 10 important features (Random Forest `feature_importances_`):
Online boarding, Inflight wifi service, Class_Business,
Type of Travel_Personal Travel, Inflight entertainment,
Type of Travel_Business travel, Seat comfort, Ease of Online booking,
Class_Eco, Flight Distance.
Note: importance means the model found these features useful for prediction;
it does NOT prove they cause satisfaction.

## 7. Final Model

**Random Forest** was selected because it has the highest F1 score (0.9563)
— it is the best balance of precision (0.9717) and recall (0.9414).
It is saved (with preprocessing info) in `models/final_model.pkl`.

## 8. KPIs

| KPI | Type | Formula | Target* | Actual | Interpretation |
|---|---|---|---|---|---|
| Passenger Satisfaction Rate | Business | Satisfied / Total × 100 | 70.00% | 43.45% | Share of passengers reported as satisfied in the dataset. Below the project target — service improvement is needed. |
| Accuracy | ML | Correct / Total predictions | 85.00% | 96.27% | 96.27% of test predictions were correct. Target met. |
| Precision | ML | TP / (TP + FP) | 80.00% | 97.17% | When the model predicts satisfied, it is right 97.17% of the time. Target met. |
| Recall | ML | TP / (TP + FN) | 80.00% | 94.14% | The model finds 94.14% of all truly satisfied passengers. Target met. |
| F1 Score | ML | 2·P·R / (P + R) | 80.00% | 95.63% | Balanced score; well above target. |

\*Targets are simple project targets, not real airline standards.

## 9. Data Product

`app.py` is a simple Streamlit app titled
"Airline Passenger Satisfaction Predictor". The user enters age, gender,
customer type, travel type, class, flight distance, departure/arrival delay
and 5 service ratings, clicks **Predict Satisfaction**, and sees
`SATISFIED` or `DISSATISFIED` plus the prediction probability. Below that,
the app shows the final model's Accuracy, Precision, Recall, F1 and the
3-model comparison table — proving the app is connected to the ML work.
It applies the SAME preprocessing (one-hot columns from training) before
predicting.

## 10. How to Run

```bash
pip install -r requirements.txt
```

```bash
python ml_model.py
```

```bash
streamlit run app.py
```
