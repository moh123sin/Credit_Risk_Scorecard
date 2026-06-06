# 💰 Credit Risk Scorecard

An end-to-end ML credit default prediction system with SHAP explainability, model comparison, and a live loan applicant scorer.

## 🔗 Live Demo
Deploy free on [Streamlit Cloud](https://streamlit.io/cloud).

## 🎯 Business Problem
Banks and lending companies need to assess the likelihood of a borrower defaulting on a loan. This project builds a full credit scoring pipeline — from raw applicant data to an explainable risk decision — mimicking a real-world lending scorecard.

## 🛠️ Tech Stack
- **Python** · Pandas · NumPy
- **Scikit-learn** — Logistic Regression, Random Forest, Gradient Boosting
- **SHAP** — model explainability (TreeExplainer)
- **Streamlit** — interactive web app
- **Matplotlib** — ROC curves, confusion matrix, SHAP plots

## 📊 Features
- **Portfolio Overview** — default rates by DTI, home ownership, loan purpose
- **Live Applicant Scorer** — enter applicant details, get credit score + decision
- **Model Comparison** — ROC curves and confusion matrix across 3 models
- **SHAP Explainability** — global feature importance + per-feature SHAP distributions

## 🧠 ML Approach
1. Feature engineering on applicant data (income, DTI, delinquencies, credit history)
2. Three models trained: Logistic Regression, Random Forest, Gradient Boosting
3. Balanced class weights to handle default class imbalance
4. ROC-AUC and Precision-Recall as evaluation metrics
5. SHAP TreeExplainer for post-hoc model explainability
6. Credit score mapping (300–850 scale) from default probability

## 🚀 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Project Structure
```
credit_risk/
├── app.py              # Main Streamlit app
├── requirements.txt
└── README.md
```

## 📈 Results
| Model | ROC-AUC |
|-------|---------|
| Logistic Regression | ~0.74 |
| Random Forest | ~0.79 |
| Gradient Boosting | ~0.81 |

## 💡 Key Insights (from SHAP)
- **Debt-to-income ratio** is the strongest predictor of default
- **Past delinquencies** sharply increase risk even at low counts
- **Credit history length** is strongly protective
- **Income** reduces risk — higher earners default less frequently
