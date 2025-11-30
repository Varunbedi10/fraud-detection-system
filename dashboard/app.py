import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pathlib
import matplotlib.pyplot as plt

# ------ PAGE CONFIG (bank/executive style) ------
st.set_page_config(
    page_title="Credit Card Fraud Risk Console",
    layout="wide"
)

# ---- Simple styling to feel like a bank dashboard ----
st.markdown(
    """
    <style>
        .main {
            background-color: #0b1120;
            color: #e5e7eb;
        }
        .stMetric {
            background-color: #020617 !important;
            padding: 12px;
            border-radius: 12px;
            border: 1px solid #1f2937;
        }
        .big-title {
            font-size: 32px;
            font-weight: 700;
            color: #facc15;
        }
        .sub-title {
            font-size: 16px;
            color: #9ca3af;
        }
        .section-title {
            font-size: 20px;
            font-weight: 600;
            margin-top: 20px;
            color: #e5e7eb;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ------ LOAD MODEL & SCALER ------
ROOT = pathlib.Path(__file__).resolve().parent.parent
model_path = ROOT / "models" / "xgb_best.pkl"
scaler_path = ROOT / "models" / "scaler.pkl"

@st.cache_resource
def load_model_and_scaler():
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_model_and_scaler()

# ------ HEADER ------
st.markdown('<div class="big-title">💳 Credit Card Fraud Risk Console</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">'
    'This tool scans a batch of credit card transactions and highlights which ones are likely to be fraud, '
    'how much money is at risk, and where attention is needed.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("---")

# ------ FILE UPLOAD ------
st.markdown('<div class="section-title">1️⃣ Upload Transactions File</div>', unsafe_allow_html=True)
st.write(
    "Upload a CSV file with transactions."
)

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

required_cols = [
    "Time","V1","V2","V3","V4","V5","V6","V7","V8","V9",
    "V10","V11","V12","V13","V14","V15","V16","V17","V18","V19",
    "V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount"
]

if uploaded_file is None:
    st.info("No file uploaded yet. Use the uploader above to start a risk scan.")
else:
    df = pd.read_csv(uploaded_file)

    # ------ VALIDATION ------
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"File is missing required columns: {missing}")
    else:
        st.success(f"File loaded successfully. {len(df)} transactions detected.")

        # ------ PREDICTION ------
        X = df[required_cols]
        X_scaled = scaler.transform(X)
        probs = model.predict_proba(X_scaled)[:, 1]
        preds = (probs > 0.5).astype(int)

        df_result = df.copy()
        df_result["Fraud_Probability"] = probs
        df_result["Fraud_Prediction"] = preds

        # Risk buckets for non-tech users
        def risk_level(p):
            if p >= 0.85:
                return "🚨 High"
            elif p >= 0.60:
                return "⚠ Medium"
            else:
                return "✅ Low"

        def suggested_action(p):
            if p >= 0.85:
                return "Block immediately / manual review"
            elif p >= 0.60:
                return "Extra verification (OTP / call customer)"
            else:
                return "Allow, but monitor"

        df_result["Risk_Level"] = df_result["Fraud_Probability"].apply(risk_level)
        df_result["Suggested_Action"] = df_result["Fraud_Probability"].apply(suggested_action)

        # ------ BUSINESS SUMMARY KPIs ------
        st.markdown('<div class="section-title">2️⃣ Business Summary</div>', unsafe_allow_html=True)

        total_tx = len(df_result)
        high_risk = df_result[df_result["Risk_Level"] == "🚨 High"]
        medium_risk = df_result[df_result["Risk_Level"] == "⚠ Medium"]

        total_high = len(high_risk)
        total_medium = len(medium_risk)

        amount_high = high_risk["Amount"].sum()
        amount_medium = medium_risk["Amount"].sum()

        avg_risk = df_result["Fraud_Probability"].mean()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Transactions Scanned", f"{total_tx}")
        col2.metric("High-Risk Transactions", f"{total_high}")
        col3.metric("Est. High-Risk Amount (₹)", f"{amount_high:,.2f}")
        col4.metric("Avg. Risk Score", f"{avg_risk:.2f}")

        # Narrative summary
        st.write("")
        st.write("**Plain-language summary:**")
        if total_high == 0 and total_medium == 0:
            st.write(
                "- No medium or high-risk transactions detected in this batch.\n"
                "- This batch appears generally safe based on the model's risk scores."
            )
        else:
            st.write(
                f"- There are **{total_high} high-risk** and **{total_medium} medium-risk** transactions.\n"
                f"- Approximately **₹{amount_high:,.2f}** is tied to high-risk transactions. "
                "These should be reviewed or blocked to avoid potential loss.\n"
                "- Medium-risk transactions may require extra verification (OTP / call)."
            )

        st.markdown("---")

        # ------ RISK DISTRIBUTION VISUALS ------
        st.markdown('<div class="section-title">3️⃣ Risk Distribution Overview</div>', unsafe_allow_html=True)

        col_a, col_b = st.columns(2)

        # Pie chart of risk levels
        with col_a:
            risk_counts = df_result["Risk_Level"].value_counts().reindex(
                ["🚨 High", "⚠ Medium", "✅ Low"], fill_value=0
            )
            fig1, ax1 = plt.subplots()
            ax1.pie(
                risk_counts.values,
                labels=risk_counts.index,
                autopct='%1.1f%%',
                startangle=90
            )
            ax1.axis('equal')
            st.pyplot(fig1)
            st.caption("Share of transactions by risk category")

        # Histogram of fraud probabilities
        with col_b:
            fig2, ax2 = plt.subplots()
            ax2.hist(df_result["Fraud_Probability"], bins=30)
            ax2.set_xlabel("Fraud probability")
            ax2.set_ylabel("Number of transactions")
            st.pyplot(fig2)
            st.caption("Distribution of risk scores across all transactions")

        st.markdown("---")

        # ------ HIGH-RISK TRANSACTIONS TABLE ------
        st.markdown('<div class="section-title">4️⃣ High-Risk Transactions (Action Required)</div>', unsafe_allow_html=True)
        if total_high == 0:
            st.info("No high-risk transactions in this batch.")
        else:
            display_cols = ["Time", "Amount", "Fraud_Probability", "Risk_Level", "Suggested_Action"]
            st.write(
                "These are the transactions the model considers highly suspicious. "
                "They are sorted by risk score so the riskiest appear first."
            )
            st.dataframe(
                high_risk[display_cols].sort_values("Fraud_Probability", ascending=False),
                use_container_width=True
            )

        # ------ ALL TRANSACTIONS WITH FLAGS ------
        st.markdown('<div class="section-title">5️⃣ Full Transaction List with Risk Flags</div>', unsafe_allow_html=True)
        display_cols_all = ["Time", "Amount", "Fraud_Probability", "Risk_Level", "Suggested_Action"]
        st.dataframe(
            df_result[display_cols_all].sort_values("Fraud_Probability", ascending=False),
            use_container_width=True
        )

        # ------ DOWNLOAD RESULTS ------
        st.markdown('<div class="section-title">6️⃣ Export Results</div>', unsafe_allow_html=True)
        st.write("You can download the full list with risk labels and suggested actions for reporting or follow-up.")
        csv_out = df_result.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Results as CSV",
            data=csv_out,
            file_name="fraud_detection_results_labeled.csv",
            mime="text/csv",
        )
