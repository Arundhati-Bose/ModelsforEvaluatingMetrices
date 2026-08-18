#Load Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    matthews_corrcoef
)
import streamlit as st


ModelName = st.selectbox(
    "Model Selection",
    ("Select a model","Logistic Regression","Decision Tree Classifier","K-Nearest Neighbor Classifier","Naive Bayes Classifi er - Gaussian or Multinomial","Random Forest")
)

#st.write("Model Selected:",ModelName)


# 1. Get the absolute path of the directory where app.py is running
# __file__ is a built-in variable that points to your app.py script location
BASE_DIR = Path(__file__).resolve().parent

# Join the base directory path with your dataset filename
DATASET_PATH = BASE_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

print(BASE_DIR)

# 1. Load the dataset (Update path to your local downloaded file)
df = pd.read_csv(DATASET_PATH)

# Force Pandas to show all content
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

#Take user entry
#ModelName = input("Enter Model Name: ")

if ModelName == "Logistic Regression" :
   tag = 1
elif ModelName == "Decision Tree Classifier" :
   tag = 2
elif ModelName == "K-Nearest Neighbor Classifier" :
   tag = 3
elif ModelName == "Naive Bayes Classifi er - Gaussian or Multinomial" :
   tag = 4
elif ModelName == "Random Forest":
   tag = 5
else:
   tag = 6
# Print 
#print(df)

# Logistic Regression
# 1. Data Cleaning & Preprocessing
# Drop non-predictive Identifier column
df = df.drop(columns=['customerID'])

# Convert TotalCharges to numeric, turning empty spaces (" ") into NaN values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill the few missing values in TotalCharges with its median
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Map Target variable 'Churn' to binary numeric values
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# One-Hot Encode all remaining text categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

# 3. Separate Features (X) and Target (y)
X = df_encoded.drop(columns=['Churn'])
y = df_encoded['Churn']

# 4. Split into Training and Testing sets (80/20 split)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 5. Feature Scaling (Highly recommended for stable Logistic Regression convergence)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Initialize and Fit the Logistic Regression Model
# Increasing max_iter ensures complete convergence during optimization
if tag == 1:
   model = LogisticRegression(max_iter=1000)
elif tag == 2:
   model = DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42)
elif tag == 3:
   model = KNeighborsClassifier(n_neighbors=11)
elif tag == 4:
   model = GaussianNB()
else:
   model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=8, 
    class_weight='balanced', 
    random_state=42
   )


model.fit(X_train_scaled, y_train)


# 7. Generate Classification Predictions
y_pred = model.predict(X_test_scaled)          # For discrete metric counts
y_prob = model.predict_proba(X_test_scaled)[:, 1] # For the AUC ROC continuous scale

# 8. Compute all 6 requested evaluation metrics
metrics = {
    "1. Accuracy": accuracy_score(y_test, y_pred),
    "2. AUC Score": roc_auc_score(y_test, y_prob),
    "3. Precision": precision_score(y_test, y_pred),
    "4. Recall": recall_score(y_test, y_pred),
    "5. F1 Score": f1_score(y_test, y_pred),
    "6. Matthews Correlation Coefficient (MCC Score)": matthews_corrcoef(y_test, y_pred)
}

# 9. Output Report
if tag == 1:
  st.write("=== Telco Churn Logistic Regression Metrics ===")
elif tag == 2:
  st.write("=== Telco Churn Decision Tree Classifier ===")
elif tag == 3:
   st.write("=== Telco Churn K-Nearest Neighbor Classifier ===")
elif tag == 4:
   st.write("=== Telco Churn Naive Bayes Classifi er - Gaussian or Multinomial ===")
elif tag == 5:
   st.write("=== Telco Churn Ensemble Model - Random Forest ===")

if tag != 6:
   for name, score in metrics.items():
      st.write(f"{name}: {score:.4f}")
