# coding: utf-8
#logistic_regression.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, \
                            f1_score, roc_auc_score, roc_curve, recall_score
import pickle

# 1. Load the dataset
df = pd.read_csv('diabetes.csv')
print("--- Dataset Info ---")
print(df.info())
print("--- Summary Statistics ---")
print(df.describe())

# 2. Check and handle missing values
print("--- Missing Values ---")
print(df.isnull().sum())

cols_with_missing = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
df[cols_with_missing] = df[cols_with_missing].replace(0, np.nan)
for col in cols_with_missing:
    df[col].fillna(df[col].median(), inplace=True)
print(df.isnull().sum())

# 3. Univariate Analysis – Histograms
df.hist(figsize=(12,10), bins=20)
plt.suptitle('Histograms of All Features')
plt.show()

# 4. Boxplots for Outlier Detection
plt.figure(figsize=(12,8))
df.boxplot()
plt.title("Boxplots of All Features")
plt.xticks(rotation=45)
plt.show()

# 5. Prepare features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 8. Train Logistic Regression Model
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

# 9. Save model and scaler as .pkl files for later use (for deployment)
with open('logisticmodel.pkl', 'wb') as file:
    pickle.dump(log_model, file)
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# 10. Make predictions & Evaluate
y_pred = log_model.predict(X_test)
y_pred_proba = log_model.predict_proba(X_test)[:,1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
rocauc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {acc}")
print(f"Precision: {prec}")
print(f"Recall: {rec}")
print(f"F1-score: {f1}")
print(f"ROC-AUC Score: {rocauc}")

# 11. Feature importance/coefficients
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': log_model.coef_[0]
}).sort_values(by='Coefficient', ascending=False)
print(coef_df)

# 12. ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC {rocauc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random chance')
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend()
plt.grid(True)
plt.show()
