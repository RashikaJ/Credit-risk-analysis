# ================== Imports ==================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)

from imblearn.over_sampling import SMOTE

# ================== Global Plot Settings ==================
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# ================== Data Loading & Cleaning ==================
df = pd.read_excel('Bank_Personal_Loan_Modelling.xlsx', sheet_name="Data")
df.drop(columns=['ID'], inplace=True)

# ================== Basic Info ==================
print(f"Shape: {df.shape}")
print(df.info())
print("Missing Values:\n", df.isnull().sum())

# ================== EDA ==================

# Target Distribution
sns.countplot(data=df, x='Personal Loan', palette='Set2')
plt.title("Distribution of Personal Loan Acceptance")
plt.show()
print(df['Personal Loan'].value_counts(normalize=True))

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Income vs Loan
sns.boxplot(data=df, x='Personal Loan', y='Income', palette='Set3')
plt.title("Income vs Personal Loan")
plt.show()

# Credit Card Spend vs Loan
sns.boxplot(data=df, x='Personal Loan', y='CCAvg', palette='Set1')
plt.title("Credit Card Spend vs Personal Loan")
plt.show()

# Binary Columns vs Loan
binary_cols = ['Education', 'Family', 'Securities Account', 'CD Account', 'Online', 'CreditCard']
for col in binary_cols:
    sns.countplot(data=df, x=col, hue='Personal Loan', palette='pastel')
    plt.title(f"{col} vs Personal Loan Acceptance")
    plt.show()

# ================== Data Prep ==================
X = df.drop('Personal Loan', axis=1)
y = df['Personal Loan']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ================== Logistic Regression ==================
logreg = LogisticRegression()
logreg.fit(X_train_scaled, y_train)

y_pred_log = logreg.predict(X_test_scaled)
y_prob_log = logreg.predict_proba(X_test_scaled)[:, 1]

print("Classification Report (Logistic Regression):\n", classification_report(y_test, y_pred_log))

cm_log = confusion_matrix(y_test, y_pred_log)
sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

roc_log = roc_auc_score(y_test, y_prob_log)
print(f"ROC-AUC (Logistic Regression): {roc_log:.4f}")

# ================== Apply SMOTE ==================
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

print("Before SMOTE:\n", y_train.value_counts())
print("After SMOTE:\n", pd.Series(y_train_resampled).value_counts())

# ================== Random Forest ==================
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)

y_pred_rf = rf_model.predict(X_test_scaled)
y_prob_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

print("Classification Report (Random Forest):\n", classification_report(y_test, y_pred_rf))

cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

roc_rf = roc_auc_score(y_test, y_prob_rf)
print(f"ROC-AUC (Random Forest): {roc_rf:.4f}")

# ================== ROC Curve Comparison ==================
fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

plt.plot(fpr_log, tpr_log, label=f'Logistic (AUC = {roc_log:.2f})', linestyle='--')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_rf:.2f})')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(True)
plt.show()

# ================== Feature Importance (Random Forest) ==================
importances = rf_model.feature_importances_
feature_names = X.columns
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

sns.barplot(x=feat_imp[:10], y=feat_imp.index[:10], palette='viridis')
plt.title("Top 10 Feature Importances - Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

# ================== Final Model Comparison Summary ==================
comparison_df = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "ROC-AUC": [roc_log, roc_rf],
    "Accuracy": [
        (y_pred_log == y_test).mean(),
        (y_pred_rf == y_test).mean()
    ]
})
print("\nModel Comparison Summary:")
print(comparison_df.round(4))
