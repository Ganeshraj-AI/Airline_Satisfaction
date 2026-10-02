# Airline Passenger Satisfaction Prediction System
# ----------------------------------------------
# ONE single ML file for the whole project (as required for CA2).
# Flow: Dataset -> Cleaning -> Preprocessing -> Train/Test Split
#       -> 3 ML Models -> Evaluation -> Comparison -> Final Model -> Save Model
#
# Run with:  python ml_model.py
# Needs: pandas, numpy, scikit-learn, matplotlib, seaborn, joblib

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")  # save plots without opening a window
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ---------------------------------------------------------------
# 0. PATHS (keep it simple: paths relative to this file)
# ---------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "airline_passenger_satisfaction.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "final_model.pkl")
CM_PATH = os.path.join(MODEL_DIR, "confusion_matrix.png")
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------
# We use pandas to read the CSV file.
df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("1. DATA INSPECTION")
print("=" * 60)
print("Shape (rows, columns):", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn names:")
print(list(df.columns))
print("\nData types:")
print(df.dtypes)
print("\nTarget column value counts:")
print(df["satisfaction"].value_counts())

# ---------------------------------------------------------------
# 2. REMOVE UNNECESSARY COLUMNS (ID columns)
# ---------------------------------------------------------------
# Columns like 'Feature 1' / 'Unnamed: 0' / 'id' are just row numbers
# or passenger IDs. They are unique for every row, so the model cannot
# learn any general pattern from them. We drop them.
for col in ["Feature 1", "Unnamed: 0", "id", "ID"]:
    if col in df.columns:
        df = df.drop(columns=[col])
        print(f"Removed ID column: {col}")

print("\nShape after removing ID columns:", df.shape)

# ---------------------------------------------------------------
# 3. HANDLE MISSING VALUES
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("2. MISSING VALUES")
print("=" * 60)
print(df.isnull().sum())  # show missing value count per column

# Rule used here (simple and standard):
#   Numerical columns -> fill with median (safe from outliers)
#   Categorical columns -> fill with mode (most common value)
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
# 'satisfaction' is the target, not a feature, so remove it from cat list
if "satisfaction" in cat_cols:
    cat_cols.remove("satisfaction")

print("\nNumerical columns:", num_cols)
print("Categorical columns:", cat_cols)

# NOTE: we fill AFTER the train/test split below using TRAIN statistics
# only, so that test data does not influence training (no data leakage).
# We just report the missing values here.

# ---------------------------------------------------------------
# 4. REMOVE DUPLICATES
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("3. DUPLICATES")
print("=" * 60)
print("Duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates()  # keep only one copy of repeated rows
print("Shape after removing duplicates:", df.shape)

# ---------------------------------------------------------------
# 5. FEATURE SELECTION  (X = features, y = target)
# ---------------------------------------------------------------
# Target column is 'satisfaction' with values:
#   'satisfied'               -> 1 (positive class)
#   'neutral or dissatisfied' -> 0 (negative class)
# We convert to 1/0 so scikit-learn can do binary classification.
print("\n" + "=" * 60)
print("4. TARGET ENCODING")
print("=" * 60)
print("Original target values:", df["satisfaction"].unique())
df["satisfaction"] = df["satisfaction"].map(
    {"satisfied": 1, "neutral or dissatisfied": 0}
)
print("Mapped target values: Satisfied = 1, Dissatisfied = 0")
print(df["satisfaction"].value_counts())

# X = all columns except target, y = target
X = df.drop(columns=["satisfaction"])
y = df["satisfaction"]

# Business KPI 1: Passenger Satisfaction Rate (from the full cleaned data)
satisfaction_rate = y.mean() * 100
print(f"\nPassenger Satisfaction Rate = {satisfaction_rate:.2f}%")
print("(Formula: Satisfied Passengers / Total Passengers x 100)")

# ---------------------------------------------------------------
# 6. TRAIN / TEST SPLIT  (80% train, 20% test)
# ---------------------------------------------------------------
# We split BEFORE preprocessing so the test set stays unseen.
# stratify=y keeps the same % of satisfied/dissatisfied in both sets.
# random_state=42 makes the split repeatable.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print("\n" + "=" * 60)
print("5. TRAIN / TEST SPLIT (80/20, random_state=42, stratify=y)")
print("=" * 60)
print("Train shape:", X_train.shape, " Test shape:", X_test.shape)

# ---------------------------------------------------------------
# 7. HANDLE MISSING VALUES USING TRAIN STATISTICS ONLY
# ---------------------------------------------------------------
# To avoid data leakage, medians/modes are computed on TRAIN data only,
# then applied to both train and test.
num_medians = X_train[num_cols].median()
cat_modes = X_train[cat_cols].mode().iloc[0]  # most common value per column

X_train = X_train.fillna({**num_medians.to_dict(), **cat_modes.to_dict()})
X_test = X_test.fillna({**num_medians.to_dict(), **cat_modes.to_dict()})
print("\nMissing values after filling - train:", X_train.isnull().sum().sum(),
      " test:", X_test.isnull().sum().sum())

# ---------------------------------------------------------------
# 8. ENCODING (handle categorical columns)
# ---------------------------------------------------------------
# Models need numbers, so text columns (Gender, Customer Type, etc.)
# are converted with pd.get_dummies() = one-hot encoding.
# Example: Gender -> Gender_Female (0/1), Gender_Male (0/1).
# We encode train, then align test to the SAME columns (fill missing with 0).
X_train = pd.get_dummies(X_train, drop_first=False)
train_columns = X_train.columns.tolist()  # save this for app.py
X_test = pd.get_dummies(X_test, drop_first=False)
X_test = X_test.reindex(columns=train_columns, fill_value=0)

print("\nTotal features after encoding:", len(train_columns))
print("Example encoded columns:", train_columns[:8])

# ---------------------------------------------------------------
# 9. TRAIN THREE MODELS
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("6. MODEL TRAINING (3 models)")
print("=" * 60)

# Model 1: Logistic Regression (simple baseline for classification)
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
print("Trained: Logistic Regression")

# Model 2: Decision Tree (tree-based rules like 'if Online boarding < 3 ...')
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
print("Trained: Decision Tree")

# Model 3: Random Forest (100 decision trees voting together)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print("Trained: Random Forest")

# ---------------------------------------------------------------
# 10. MODEL EVALUATION (Accuracy, Precision, Recall, F1)
# ---------------------------------------------------------------
# Accuracy  = correct / total
# Precision = TP / (TP + FP)  -> when we say satisfied, how often right?
# Recall    = TP / (TP + FN)  -> of all truly satisfied, how many found?
# F1        = 2*P*R/(P+R)     -> balance of precision and recall


def get_scores(model, X_t, y_t):
    pred = model.predict(X_t)
    return {
        "accuracy": accuracy_score(y_t, pred),
        "precision": precision_score(y_t, pred),
        "recall": recall_score(y_t, pred),
        "f1": f1_score(y_t, pred),
    }


results = {
    "Logistic Regression": get_scores(lr, X_test, y_test),
    "Decision Tree": get_scores(dt, X_test, y_test),
    "Random Forest": get_scores(rf, X_test, y_test),
}

print("\n" + "=" * 60)
print("7. MODEL COMPARISON")
print("=" * 60)
print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
print("-" * 60)
for name, s in results.items():
    print(f"{name:<20} {s['accuracy']:<10.4f} {s['precision']:<10.4f} "
          f"{s['recall']:<10.4f} {s['f1']:<10.4f}")

# ---------------------------------------------------------------
# 11. MODEL SELECTION (pick best by F1 score)
# ---------------------------------------------------------------
# F1 is preferred because we care about BOTH missing satisfied
# passengers (recall) AND wrong satisfied predictions (precision).
best_name = max(results, key=lambda k: results[k]["f1"])
best_model = {"Logistic Regression": lr, "Decision Tree": dt,
              "Random Forest": rf}[best_name]
best = results[best_name]
print("\n" + "=" * 60)
print(f"8. FINAL MODEL: {best_name} (highest F1 = {best['f1']:.4f})")
print("=" * 60)

# ---------------------------------------------------------------
# 12. CONFUSION MATRIX FOR FINAL MODEL
# ---------------------------------------------------------------
# Rows = actual, Columns = predicted.
# [[TN, FP], [FN, TP]] for classes [0=Dissatisfied, 1=Satisfied]
y_pred_final = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_final)
print("\nConfusion Matrix (rows=actual 0/1, cols=predicted 0/1):")
print(cm)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Dissatisfied", "Satisfied"],
            yticklabels=["Dissatisfied", "Satisfied"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig(CM_PATH)
print(f"Confusion matrix image saved to: {CM_PATH}")

# ---------------------------------------------------------------
# 13. FEATURE IMPORTANCE (top 10)
# ---------------------------------------------------------------
# Shows which features the model found most useful for prediction.
# This is NOT proof of cause - only that they help prediction.
print("\n" + "=" * 60)
print("9. FEATURE IMPORTANCE (top 10)")
print("=" * 60)
if hasattr(best_model, "feature_importances_"):  # trees / forest
    importances = pd.Series(best_model.feature_importances_,
                            index=train_columns).sort_values(ascending=False)
    print(importances.head(10))
elif hasattr(best_model, "coef_"):  # logistic regression
    importances = pd.Series(abs(best_model.coef_[0]),
                            index=train_columns).sort_values(ascending=False)
    print("(Using |coefficient| as importance for Logistic Regression)")
    print(importances.head(10))

# ---------------------------------------------------------------
# 14. KPI TABLE (5 KPIs, actual values from this run)
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("10. KPI TABLE")
print("=" * 60)
print(f"{'KPI':<28} {'Type':<10} {'Target*':<10} {'Actual':<10}")
print("-" * 60)
print(f"{'Satisfaction Rate':<28} {'Business':<10} {'70.00%':<10} "
      f"{satisfaction_rate:<10.2f}")
print(f"{'Accuracy':<28} {'ML':<10} {'85.00%':<10} {best['accuracy'] * 100:<10.2f}")
print(f"{'Precision':<28} {'ML':<10} {'80.00%':<10} {best['precision'] * 100:<10.2f}")
print(f"{'Recall':<28} {'ML':<10} {'80.00%':<10} {best['recall'] * 100:<10.2f}")
print(f"{'F1 Score':<28} {'ML':<10} {'80.00%':<10} {best['f1'] * 100:<10.2f}")
print("\n*Targets are simple project targets, not real airline standards.")

# ---------------------------------------------------------------
# 15. SAVE FINAL MODEL + EVERYTHING app.py NEEDS
# ---------------------------------------------------------------
# app.py needs the same columns, fill values and metrics, so we save
# them together with the model in ONE file.
bundle = {
    "model": best_model,
    "model_name": best_name,
    "columns": train_columns,              # one-hot columns from training
    "num_medians": num_medians.to_dict(),  # for filling missing numbers
    "cat_modes": cat_modes.to_dict(),      # for filling missing categories
    "num_cols": num_cols,
    "cat_cols": cat_cols,
    "metrics": best,                       # final model metrics
    "comparison": results,                 # all 3 models
    "satisfaction_rate": satisfaction_rate,
}
joblib.dump(bundle, MODEL_PATH)
print(f"\nFinal model + preprocessing saved to: {MODEL_PATH}")
print("\nDONE. Next step: streamlit run app.py")
