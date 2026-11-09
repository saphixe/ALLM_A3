# ===========================
# STEP 0: Train & Save Models
# ===========================
import pandas as pd
import numpy as np
import uuid
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

os.makedirs("model", exist_ok=True)
os.makedirs("data/test", exist_ok=True)

def split_data(X, y, disease):
       #reset indices to avoid misalignment issues
       X = X.reset_index(drop=True) 
       y = pd.Series(np.asarray(y)).reset_index(drop=True)

       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

       joblib.dump( {"X": X_test.reset_index(drop=True),
              "y": y_test.reset_index(drop=True)},
              f"data/test/{disease}_test.pkl")

       return X_train, y_train

# Diabetes Model
print("Training Diabetes Model...")
diabetes = pd.read_csv("data/pima-indians-diabetes.data.csv",
names=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
       "DiabetesPedigreeFunction", "Age", "Outcome"])
X_d_train, y_d_train = diabetes.drop("Outcome", axis=1), diabetes["Outcome"]

X_d_train, y_d_train = split_data(X_d_train, y_d_train, "diabetes")

clf_diabetes = RandomForestClassifier().fit(X_d_train, y_d_train)
joblib.dump(clf_diabetes, "model/diabetes.pkl")

# Breast Cancer Model
print("Training Breast Cancer Model...")
breast = load_breast_cancer()
X_b_train = pd.DataFrame(breast.data, columns=breast.feature_names)
y_b_train = breast.target  # 0 = malignant, 1 = benign

X_b_train, y_b_train = split_data(X_b_train, y_b_train, "breast_cancer")

clf_breast = RandomForestClassifier().fit(X_b_train, y_b_train)
joblib.dump(clf_breast, "model/breast.pkl")

# Heart Model
print("Training Heart Model...")
heart = pd.read_csv("data/heart.csv")
X_h_train, y_h_train = heart.drop("target", axis=1), heart["target"]

X_h_train, y_h_train = split_data(X_h_train, y_h_train, "heart")

clf_heart = RandomForestClassifier().fit(X_h_train, y_h_train)
joblib.dump(clf_heart, "model/heart.pkl")

print("Models trained and saved in 'model/' folder.")