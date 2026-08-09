import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from config.env import load_repo_env
from monitor.cost_drift import cost_weighted_drift, kl_per_feature


load_repo_env()

st.set_page_config(page_title="Cost-Weighted Fraud Monitor", layout="wide")
st.title("Cost-Weighted Fraud and Drift Console")

scorer_url = os.getenv("SCORER_URL", "http://localhost:8080")
demo_dir = Path(os.getenv("DEMO_DATA_DIR", "outputs/demo"))
baseline_path = demo_dir / "baseline_window.parquet"
production_path = demo_dir / "production_window.parquet"
importance_path = demo_dir / "shap_importance.json"

col1, col2 = st.columns(2)

with col1:
    st.subheader("Score a transaction")
    amount = st.number_input("Amount ($)", value=120.0)
    card1 = st.number_input("card1", value=13926, step=1)
    if st.button("Score"):
        resp = requests.post(
            f"{scorer_url}/predict",
            json={"TransactionAmt": amount, "card1": int(card1), "ProductCD": "W"},
            timeout=30,
        )
        result = resp.json()
        st.metric("Fraud probability", f"{result['fraud_prob']:.1%}")
        st.write("Flagged" if result["flag"] else "Not flagged")

with col2:
    st.subheader("Drift leaderboard")
    if baseline_path.exists() and production_path.exists() and importance_path.exists():
        baseline = pd.read_parquet(baseline_path)
        production = pd.read_parquet(production_path)
        importance = json.loads(importance_path.read_text(encoding="utf-8"))
        kl = kl_per_feature(baseline, production)
        ranked = cost_weighted_drift(kl, importance, p_fraud=0.013)
        st.bar_chart(ranked.set_index("feature")["cost_drift_usd"])
        st.dataframe(ranked.head(10), use_container_width=True)
    else:
        st.info("Create demo parquet files under outputs/demo to render the drift view.")
