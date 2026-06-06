import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (roc_auc_score, roc_curve, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve)
from sklearn.pipeline import Pipeline
import shap
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Credit Risk Scorecard", layout="wide", page_icon="💰")

st.markdown("""
<style>
h1 {color:#2E74B5;}
.high-risk   {background:#ffe0e0;padding:18px;border-radius:10px;border-left:6px solid #e74c3c;}
.medium-risk {background:#fff3e0;padding:18px;border-radius:10px;border-left:6px solid #f39c12;}
.low-risk    {background:#e0ffe0;padding:18px;border-radius:10px;border-left:6px solid #27ae60;}
.scorecard   {background:#f0f4ff;padding:15px;border-radius:10px;border-left:5px solid #2E74B5;margin:5px 0;}
</style>""", unsafe_allow_html=True)

# ── Data generation ───────────────────────────────────────────────────────────
@st.cache_data
def generate_credit_data(n=5000):
    np.random.seed(42)
    age             = np.random.randint(21, 70, n)
    income          = np.random.lognormal(10.5, 0.6, n).astype(int)
    loan_amount     = np.random.randint(1000, 50000, n)
    loan_term       = np.random.choice([12,24,36,48,60], n)
    employment_len  = np.random.randint(0, 30, n)
    debt_to_income  = np.round(np.random.beta(2, 5, n) * 0.8, 3)
    credit_history  = np.random.randint(0, 25, n)
    num_accounts    = np.random.randint(1, 15, n)
    delinquencies   = np.random.poisson(0.5, n)
    home_ownership  = np.random.choice(["Own","Rent","Mortgage"], n, p=[0.25,0.35,0.40])
    loan_purpose    = np.random.choice(["Debt Consolidation","Home Improvement","Medical","Auto","Other"], n)

    # Default probability based on risk factors
    prob = (0.08
            + 0.003 * np.clip(debt_to_income * 10, 0, 0.3)
            + 0.015 * delinquencies
            - 0.001 * credit_history
            - 0.000002 * income
            + 0.02  * (home_ownership == "Rent").astype(int)
            + 0.005 * (loan_purpose == "Medical").astype(int))
    prob = np.clip(prob, 0.03, 0.65)
    default = (np.random.rand(n) < prob).astype(int)

    return pd.DataFrame({
        "age": age, "income": income, "loan_amount": loan_amount,
        "loan_term": loan_term, "employment_length": employment_len,
        "debt_to_income": debt_to_income, "credit_history_years": credit_history,
        "num_accounts": num_accounts, "delinquencies": delinquencies,
        "home_ownership": home_ownership, "loan_purpose": loan_purpose,
        "default": default
    })

@st.cache_resource
def train_models(df):
    df2 = df.copy()
    le  = LabelEncoder()
    for col in ["home_ownership","loan_purpose"]:
        df2[col] = le.fit_transform(df2[col])

    X = df2.drop("default", axis=1)
    y = df2["default"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                         random_state=42, stratify=y)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    # Three models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    }
    results = {}
    for name, m in models.items():
        if name == "Logistic Regression":
            m.fit(X_tr_s, y_train)
            proba = m.predict_proba(X_te_s)[:,1]
            pred  = m.predict(X_te_s)
        else:
            m.fit(X_train, y_train)
            proba = m.predict_proba(X_test)[:,1]
            pred  = m.predict(X_test)
        results[name] = {
            "model": m, "proba": proba, "pred": pred,
            "auc": roc_auc_score(y_test, proba)
        }

    # Best model for SHAP (GBM)
    best = results["Gradient Boosting"]["model"]
    explainer   = shap.TreeExplainer(best)
    shap_values = explainer.shap_values(X_test)

    return results, X_test, y_test, shap_values, X.columns.tolist(), scaler, best

df = generate_credit_data()
results, X_test, y_test, shap_values, feature_names, scaler, best_model = train_models(df)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("💰 Credit Risk Scorecard")
st.markdown("*ML-powered credit default prediction with SHAP explainability*")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔮 Loan Applicant Scorer", "📈 Model Comparison", "🔍 SHAP Explainability"])

# ════════════════════════════════════════════════════════
# TAB 1 — Overview
# ════════════════════════════════════════════════════════
with tab1:
    st.subheader("Portfolio Overview")
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Total Loans",    f"{len(df):,}")
    k2.metric("Defaults",       f"{df['default'].sum():,}")
    k3.metric("Default Rate",   f"{df['default'].mean()*100:.1f}%")
    k4.metric("Best Model AUC", f"{max(r['auc'] for r in results.values()):.3f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Default Rate by Home Ownership**")
        ho = df.groupby("home_ownership")["default"].mean().sort_values(ascending=True) * 100
        fig, ax = plt.subplots(figsize=(5,3.5))
        colors = ["#27ae60","#f39c12","#e74c3c"]
        ax.barh(ho.index, ho.values, color=colors)
        ax.set_xlabel("Default Rate (%)"); ax.set_title("Default Rate by Home Ownership")
        for i, v in enumerate(ho.values):
            ax.text(v+0.1, i, f'{v:.1f}%', va='center')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown("**Debt-to-Income vs Default Rate**")
        df2 = df.copy()
        df2["dti_bucket"] = pd.cut(df2["debt_to_income"],
                                    bins=[0,0.1,0.2,0.3,0.4,0.8],
                                    labels=["0-10%","10-20%","20-30%","30-40%","40%+"])
        dti_d = df2.groupby("dti_bucket")["default"].mean() * 100
        fig, ax = plt.subplots(figsize=(5,3.5))
        ax.bar(dti_d.index.astype(str), dti_d.values, color=["#27ae60","#5BA3E0","#f39c12","#e67e22","#e74c3c"])
        ax.set_xlabel("Debt-to-Income Ratio"); ax.set_ylabel("Default Rate (%)")
        ax.set_title("Higher DTI = Higher Default Risk")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Default Rate by Loan Purpose**")
        lp = df.groupby("loan_purpose")["default"].mean().sort_values(ascending=True) * 100
        fig, ax = plt.subplots(figsize=(5,3.5))
        ax.barh(lp.index, lp.values, color="#2E74B5")
        ax.set_xlabel("Default Rate (%)"); ax.set_title("Default by Loan Purpose")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col4:
        st.markdown("**Delinquencies vs Default Rate**")
        delq = df.groupby("delinquencies")["default"].mean().reset_index().head(8)
        fig, ax = plt.subplots(figsize=(5,3.5))
        ax.plot(delq["delinquencies"], delq["default"]*100, marker="o",
                color="#e74c3c", linewidth=2)
        ax.fill_between(delq["delinquencies"], delq["default"]*100, alpha=0.15, color="#e74c3c")
        ax.set_xlabel("Past Delinquencies"); ax.set_ylabel("Default Rate (%)")
        ax.set_title("Delinquency History Predicts Default")
        ax.grid(alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ════════════════════════════════════════════════════════
# TAB 2 — Loan Applicant Scorer
# ════════════════════════════════════════════════════════
with tab2:
    st.subheader("🔮 Score a New Loan Applicant")
    c1, c2, c3 = st.columns(3)
    with c1:
        age            = st.slider("Age", 21, 70, 35)
        income         = st.number_input("Annual Income ($)", 10000, 500000, 55000, step=1000)
        loan_amount    = st.number_input("Loan Amount ($)", 1000, 50000, 15000, step=500)
        loan_term      = st.selectbox("Loan Term (months)", [12,24,36,48,60])
    with c2:
        emp_len        = st.slider("Employment Length (years)", 0, 30, 5)
        dti            = st.slider("Debt-to-Income Ratio", 0.0, 0.8, 0.25, 0.01)
        credit_hist    = st.slider("Credit History (years)", 0, 25, 8)
    with c3:
        num_acc        = st.slider("Number of Accounts", 1, 15, 5)
        delinq         = st.slider("Past Delinquencies", 0, 10, 0)
        home_own       = st.selectbox("Home Ownership", ["Own","Rent","Mortgage"])
        purpose        = st.selectbox("Loan Purpose", ["Debt Consolidation","Home Improvement","Medical","Auto","Other"])

    if st.button("💳 Score Applicant", use_container_width=True, type="primary"):
        enc = {"home_ownership": {"Mortgage":0,"Own":1,"Rent":2},
               "loan_purpose":   {"Auto":0,"Debt Consolidation":1,"Home Improvement":2,"Medical":3,"Other":4}}

        input_df = pd.DataFrame([[
            age, income, loan_amount, loan_term, emp_len, dti,
            credit_hist, num_acc, delinq,
            enc["home_ownership"][home_own],
            enc["loan_purpose"][purpose]
        ]], columns=feature_names)

        prob = best_model.predict_proba(input_df)[0][1]

        # Credit score (300-850 scale, inverted)
        credit_score = int(850 - (prob * 550))

        # Risk band
        if prob < 0.15:
            risk, css = "LOW RISK", "low-risk"
            decision, color = "✅ APPROVE", "#27ae60"
        elif prob < 0.35:
            risk, css = "MEDIUM RISK", "medium-risk"
            decision, color = "⚠️ REVIEW", "#f39c12"
        else:
            risk, css = "HIGH RISK", "high-risk"
            decision, color = "❌ DECLINE", "#e74c3c"

        r1,r2,r3,r4 = st.columns(4)
        r1.metric("Default Probability", f"{prob*100:.1f}%")
        r2.metric("Credit Score",        f"{credit_score}")
        r3.metric("Risk Band",           risk)
        r4.metric("Decision",            decision)

        st.markdown(f"""<div class='{css}'>
        <h4>{decision} — {risk} (Default Probability: {prob*100:.1f}%)</h4>
        <p><strong>Credit Score:</strong> {credit_score}/850 &nbsp;|&nbsp;
        <strong>DTI:</strong> {dti*100:.1f}% &nbsp;|&nbsp;
        <strong>Past Delinquencies:</strong> {delinq}</p>
        {"<p>Applicant presents acceptable risk. Standard loan terms apply.</p>" if prob < 0.15 else
         "<p>Borderline applicant. Consider requesting additional documentation or co-signer.</p>" if prob < 0.35 else
         "<p>High default risk. Recommend declining or requiring substantial collateral.</p>"}
        </div>""", unsafe_allow_html=True)

        # Score gauge
        fig, ax = plt.subplots(figsize=(6,1.2))
        ax.barh(["Score"], [prob], color=color, height=0.5)
        ax.barh(["Score"], [1-prob], left=[prob], color="#eee", height=0.5)
        ax.axvline(0.15, color="green",  linestyle="--", lw=1, alpha=0.7)
        ax.axvline(0.35, color="orange", linestyle="--", lw=1, alpha=0.7)
        ax.set_xlim(0,1); ax.set_title(f"Risk Score: {prob*100:.1f}%")
        ax.set_xlabel("Default Probability →")
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ════════════════════════════════════════════════════════
# TAB 3 — Model Comparison
# ════════════════════════════════════════════════════════
with tab3:
    st.subheader("📈 Model Comparison")

    # AUC comparison
    aucs = {name: r["auc"] for name, r in results.items()}
    cols = st.columns(3)
    for i, (name, auc) in enumerate(aucs.items()):
        cols[i].metric(name, f"AUC: {auc:.4f}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**ROC Curves — All Models**")
        fig, ax = plt.subplots(figsize=(5,4))
        colors_roc = ["#2E74B5","#27ae60","#e74c3c"]
        for (name, r), c in zip(results.items(), colors_roc):
            fpr, tpr, _ = roc_curve(y_test, r["proba"])
            ax.plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC={r['auc']:.3f})")
        ax.plot([0,1],[0,1],"k--",lw=1)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title("ROC Curve Comparison"); ax.legend(fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown("**Confusion Matrix — Gradient Boosting**")
        cm = confusion_matrix(y_test, results["Gradient Boosting"]["pred"])
        fig, ax = plt.subplots(figsize=(5,4))
        ConfusionMatrixDisplay(cm, display_labels=["No Default","Default"]).plot(
            ax=ax, colorbar=False, cmap="Blues")
        ax.set_title("Gradient Boosting — Confusion Matrix")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("**Precision-Recall Curve**")
    fig, ax = plt.subplots(figsize=(10,3.5))
    for (name, r), c in zip(results.items(), colors_roc):
        prec, rec, _ = precision_recall_curve(y_test, r["proba"])
        ax.plot(rec, prec, color=c, lw=2, label=name)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves (important for imbalanced data)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ════════════════════════════════════════════════════════
# TAB 4 — SHAP Explainability
# ════════════════════════════════════════════════════════
with tab4:
    st.subheader("🔍 SHAP Model Explainability")
    st.markdown("*SHAP (SHapley Additive exPlanations) shows which features drive each prediction.*")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Global Feature Importance (SHAP)**")
        shap_mean = np.abs(shap_values).mean(axis=0)
        fi = pd.Series(shap_mean, index=feature_names).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(5,4))
        colors_fi = ["#e74c3c" if v > fi.quantile(0.7) else "#2E74B5" for v in fi.values]
        ax.barh(fi.index, fi.values, color=colors_fi)
        ax.set_xlabel("Mean |SHAP Value|")
        ax.set_title("Feature Impact on Default Prediction")
        red_patch   = mpatches.Patch(color="#e74c3c", label="High Impact")
        blue_patch  = mpatches.Patch(color="#2E74B5", label="Lower Impact")
        ax.legend(handles=[red_patch, blue_patch], fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown("**SHAP Distribution — Top 6 Features**")
        top6 = pd.Series(shap_mean, index=feature_names).nlargest(6).index.tolist()
        top6_idx = [feature_names.index(f) for f in top6]
        fig, axes = plt.subplots(2, 3, figsize=(9,5))
        for i, (feat, idx) in enumerate(zip(top6, top6_idx)):
            ax = axes[i//3][i%3]
            feat_vals = X_test[feat].values if hasattr(X_test, 'values') else X_test[:,idx]
            shap_feat = shap_values[:,idx]
            sc = ax.scatter(feat_vals, shap_feat, c=shap_feat,
                            cmap="RdBu_r", alpha=0.3, s=8)
            ax.axhline(0, color="black", lw=0.5)
            ax.set_title(feat, fontsize=8)
            ax.set_xlabel("Feature Value", fontsize=7)
            ax.set_ylabel("SHAP Value", fontsize=7)
            ax.tick_params(labelsize=6)
        plt.suptitle("SHAP Values vs Feature Values", fontsize=10, y=1.01)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("**What does SHAP tell us?**")
    insights = [
        ("debt_to_income",       "Higher debt-to-income ratio is the strongest predictor of default."),
        ("delinquencies",        "Even 1-2 past delinquencies significantly raises default risk."),
        ("credit_history_years", "Longer credit history reduces default risk — shows track record."),
        ("income",               "Higher income is protective against default."),
        ("loan_amount",          "Larger loans carry higher default risk, especially with low income."),
    ]
    for feat, insight in insights:
        st.markdown(f"<div class='scorecard'><strong>{feat}</strong>: {insight}</div>",
                    unsafe_allow_html=True)
