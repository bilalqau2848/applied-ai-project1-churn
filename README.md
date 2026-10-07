# Customer Churn Prediction
### ML-Powered Customer Analytics | Week 1: Exploratory Data Analysis

This repository is part of Project 1 for Introduction to Applied AI (MPhil 2nd Semester, QAU). 
The goal is to predict customer churn for a telecom company using Machine Learning.

### 📊 Dataset
- **Source:** [Telco Customer Churn - Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 customers, 21 features
- **Target Variable:** `Churn` (Yes/No)
- **Features Include:** Demographics (gender, SeniorCitizen, Partner), Services (InternetService, PhoneService, Streaming), Account Info (Contract, tenure, MonthlyCharges, TotalCharges, PaymentMethod)

### 🔍 Week 1 - EDA Key Findings

**Overall Churn Rate:** 26.54%

**High-Risk Customer Profile:**
1.  **Contract:** Month-to-month contract customers have ~42% churn rate, compared to only ~3% for Two-year contract customers.
2.  **Tenure:** New customers (0-12 months) are most likely to churn. Churned customers have significantly lower average tenure.
3.  **Internet Service:** Fiber optic customers have the highest churn rate (~42%) vs DSL (~19%).
4.  **Payment Method:** Customers using Electronic Check have the highest churn rate (~45%).
5.  **Charges:** Higher MonthlyCharges are strongly associated with higher churn.
6.  **TotalCharges:** After fixing data type (11 missing values), shows strong correlation with tenure.

**Patterns Observed:**
- Tenure and TotalCharges are highly correlated (0.82) - logical, longer stay = more paid.
- Customers with PaperlessBilling and no TechSupport are more likely to churn.

### 🛠️ Setup

**Option 1: View on Kaggle (Recommended)**
Open the notebook directly on Kaggle - dataset is pre-attached.

**Option 2: Run Locally**
```bash
pip install pandas numpy matplotlib seaborn

# Download dataset from Kaggle link above and update path
# df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
