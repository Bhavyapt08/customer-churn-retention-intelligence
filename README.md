# Customer Churn & Retention Intelligence

This project focuses on predicting customer churn using customer purchase behaviour from the Online Retail II dataset.

The main goal is to identify customers who are likely to churn, understand the factors contributing to churn, and assign retention actions based on customer risk.

## Project Workflow

The project is divided into the following steps:

1. Data Understanding and EDA
2. Data Cleaning and Behavioural Analysis
3. Feature Engineering
4. Model Training and Evaluation
5. Model Explainability using SHAP
6. Customer Risk Scoring
7. Retention Priority and Action Assignment
8. Flask API
9. Power BI Dashboard

## Dataset

Dataset: Online Retail II

The original dataset contains transaction-level retail data.

The transaction data was cleaned and converted into customer-level features for churn prediction.

## Features

The following customer behaviour features were created:

- RecencyDays
- PurchaseFrequency
- MonetaryValue
- AverageOrderValue
- CustomerTenureDays
- UniqueProducts
- AverageQuantityPerOrder
- MedianPurchaseGap
- MeanPurchaseGap
- ReturnTransactionCount
- ReturnedQuantity
- ReturnedValue
- IsRepeatCustomer
- ReturnTransactionRate
- ProductsPerOrder
- PurchaseActivityRate

## Churn Definition

Customer churn was created based on whether the customer returned within the defined 180-day period.

Final customer distribution:

- Retained: 2,577
- Churned: 2,402

Total customers: 4,979

## Models

The following models were trained and compared:

- Logistic Regression
- Random Forest
- XGBoost
- Tuned XGBoost

### Model Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.727 | 0.709 | 0.735 | 0.722 | 0.799 | 0.766 |
| Random Forest | 0.712 | 0.692 | 0.725 | 0.708 | 0.793 | 0.754 |
| XGBoost | 0.720 | 0.698 | 0.738 | 0.717 | 0.793 | 0.749 |
| Tuned XGBoost | 0.731 | 0.703 | 0.765 | 0.733 | 0.803 | 0.768 |

Tuned XGBoost was used as the final model.

## SHAP Explainability

SHAP was used to understand which features contribute most to the churn predictions.

Top features:

- RecencyDays
- PurchaseActivityRate
- MonetaryValue
- PurchaseFrequency
- UniqueProducts

RecencyDays had the highest SHAP importance.

## Risk Segmentation

Based on predicted churn probability, customers were divided into:

- Low
- Medium
- High
- Critical

## Retention Priority

Customers were also assigned retention priorities:

- Monitor
- Medium Priority
- High Priority
- Immediate Action

Retention actions were assigned based on the customer's churn risk and priority.

## Flask API

A Flask API was created to make churn predictions using the trained model.

Endpoints:

- `/health`
- `/features`
- `/predict`

The prediction API returns:

- Churn probability
- Risk segment
- Retention priority
- Recommended action

## Power BI Dashboard

A Power BI dashboard was created for churn and retention analysis.

The dashboard includes:

- Total customers
- Actual churn rate
- Average churn probability
- High and critical risk customers
- Immediate action customers
- Priority customer value
- Risk segment distribution
- Retention priority distribution
- SHAP feature importance
- Recommended retention actions

## Live API Deployment

The Flask prediction API is deployed on Render and can be accessed at:

https://customer-churn-retention-intelligence.onrender.com

### API Endpoints

- `GET /` - Check whether the API is running
- `GET /health` - Check API health and model loading status
- `GET /features` - View the 16 features required by the model
- `POST /predict` - Generate churn probability, risk segment, retention priority, and recommended action

### Example Prediction Response

```json
{
  "churn_probability": 0.4281,
  "customer_id": null,
  "recommended_action": "Continue monitoring and regular customer engagement",
  "retention_priority": "Monitor",
  "risk_segment": "Medium"
}
```

## Project Structure

```text
customer-churn-retention-intelligence/
│
├── api/
├── dashboard/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── reports/
│   └── figures/
├── src/
└── README.md
```
