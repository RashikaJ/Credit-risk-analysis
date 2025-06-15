# Credit Risk Modeling for Personal Loan Acceptance Prediction

This project focuses on developing a supervised machine learning pipeline to predict whether a banking customer is likely to accept a personal loan offer, using real-world financial and demographic attributes. The model aims to assist financial institutions in improving lead targeting, reducing campaign costs, and optimizing loan marketing efforts.

## 💼 Business Context

Banks regularly promote personal loans as part of their cross-selling strategies. Accurately identifying high-probability responders is critical to:
- Increase marketing ROI
- Reduce customer churn
- Optimize customer engagement strategies

This model provides a scalable solution to segment customers based on their likelihood to accept personal loans.

---

## 📊 Dataset Overview

- **Source:** [Kaggle - Bank Loan Modelling Dataset](https://www.kaggle.com/datasets/itsmesunil/bank-loan-modelling)
- **Records:** 5,000 customers
- **Features:** Age, Education Level, Family Size, Income, Mortgage, Account Holdings, Online Behavior
- **Target:** `Personal Loan` (Binary: 1 = Accepted, 0 = Rejected)

---

## 🧠 Approach

1. **Exploratory Data Analysis (EDA):**
   - Target distribution, correlations, credit usage patterns
   - Customer behavior insights by education level and account type

2. **Preprocessing:**
   - Standard scaling of numerical features
   - SMOTE to address class imbalance

3. **Modeling:**
   - Logistic Regression (baseline)
   - Random Forest with hyperparameter tuning (final model)

4. **Evaluation:**
   - Confusion Matrix
   - ROC-AUC Curve
   - Feature Importance

---

## 📈 Results

| Metric            | Logistic Regression | Random Forest + SMOTE |
|-------------------|---------------------|------------------------|
| Accuracy          | 95%                 | **98.6%**              |
| ROC-AUC Score     | 0.9632              | **0.9983**             |
| Recall (Class 1)  | 66%                 | **95%**                |
| Precision (Class 1) | 80%               | **91%**                |

---

### 🔍 Key Feature Insights

- **Avg. Credit Card Spending (`CCAvg`)** and **Income** are strong predictors of loan acceptance.
- Customers with a **CD Account**, **higher education level**, or **mortgage** show increased response likelihood.

---

## 🧰 Tech Stack

- **Languages:** Python (pandas, NumPy, scikit-learn)
- **Visualization:** Seaborn, Matplotlib
- **Resampling:** SMOTE from imbalanced-learn
- **Modeling:** Logistic Regression, Random Forest
- **Evaluation:** ROC, F1-score, Confusion Matrix

---

## 📁 Folder Structure
```
credit-risk-analysis/
├── data/
├── notebooks/
│ └── credit_risk_model.ipynb
├── outputs/
│ ├── confusion_matrix_rf.png
│ ├── roc_curve_rf.png
│ └── feature_importance_rf.png
└── README.md
```

---
## 💡 Future Enhancements

- Build a Streamlit dashboard for interactive prediction
- Integrate XGBoost or LightGBM for deeper ensemble modeling
- Deploy the model as a REST API for CRM system integration

---

## 👤 Author

**Rashika Jain**  
Data Analyst 

📧 rashika0818@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/rashika-j/)
