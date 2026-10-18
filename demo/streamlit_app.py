"""Streamlit console for transaction scoring and local feature drift review."""

from __future__ import annotations

import json
import math
import os
from numbers import Real
from pathlib import Path
import sys

import pandas as pd
import requests
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.env import load_repo_env
from monitor.cost_drift import cost_weighted_drift, kl_per_feature


load_repo_env()

st.set_page_config(page_title="Cost-Weighted Fraud Monitor", layout="wide")
st.title("Cost-Weighted Fraud and Drift Console")
st.caption("Score a representative transaction and review baseline-to-production feature drift.")

DEFAULT_TEST_DATA = REPO_ROOT / "outputs" / "preprocessing" / "prepared" / "test_prepared.csv"
NON_FEATURE_COLUMNS = {"TransactionID", "TransactionDT", "isFraud"}


@st.cache_data(show_spinner="Loading sample rows from prepared test data…")
def load_test_sample(path_text: str, row_limit: int) -> pd.DataFrame:
    path = Path(path_text).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Prepared test data not found: {path}")
    header = pd.read_csv(path, nrows=0).columns
    usecols = [name for name in header if name not in NON_FEATURE_COLUMNS]
    return pd.read_csv(path, usecols=usecols, nrows=row_limit)

with st.sidebar:
    st.header("Console settings")
    scorer_url = st.text_input(
        "Scorer URL",
        value=os.getenv("SCORER_URL", "http://localhost:8080"),
        help="Base URL for the FastAPI scorer; the console calls /predict.",
    ).strip().rstrip("/")
    if st.button("Check scorer connection", disabled=not scorer_url):
        try:
            health_response = requests.get(f"{scorer_url}/health", timeout=10)
            health_response.raise_for_status()
            health = health_response.json()
            st.success(
                f"Scorer is healthy: {health.get('model_feature_count', '?')} model features, "
                f"threshold {health.get('threshold', '?')}"
            )
        except requests.RequestException as exc:
            st.error(f"Cannot reach scorer at {scorer_url}/health: {exc}")
        except (ValueError, TypeError) as exc:
            st.error(f"Scorer returned an invalid health response: {exc}")

    test_data_value = st.text_input(
        "Prepared test CSV",
        value=os.getenv("TEST_DATA_CSV", str(DEFAULT_TEST_DATA)),
    )
    test_data_path = Path(test_data_value).expanduser()
    if not test_data_path.is_absolute():
        test_data_path = REPO_ROOT / test_data_path
    sample_rows = int(st.number_input("Rows to load for scoring", min_value=1, max_value=10000, value=1000, step=100))
    demo_dir_value = st.text_input("Drift data directory", value=os.getenv("DEMO_DATA_DIR", "outputs/demo"))
    demo_dir = Path(demo_dir_value).expanduser()
    if not demo_dir.is_absolute():
        demo_dir = REPO_ROOT / demo_dir

    st.subheader("Drift assumptions")
    bins = int(st.number_input("Histogram bins", min_value=1, max_value=500, value=50, step=1))
    p_fraud = st.number_input("Estimated fraud rate", min_value=0.0, max_value=1.0, value=0.013, step=0.001, format="%.3f")
    c_fn = st.number_input("False negative cost ($)", min_value=0.0, value=420.0, step=10.0)
    c_fp = st.number_input("False positive cost ($)", min_value=0.0, value=4.0, step=1.0)
    top_n = int(st.number_input("Features to chart", min_value=1, max_value=50, value=10, step=1))

baseline_path = demo_dir / "baseline_window.parquet"
production_path = demo_dir / "production_window.parquet"
importance_path = demo_dir / "shap_importance.json"

score_col, drift_col = st.columns(2)

with score_col:
    st.subheader("Score a prepared test transaction")
    st.caption(f"Test data: {test_data_path}")
    try:
        test_sample = load_test_sample(str(test_data_path), sample_rows)
        if test_sample.empty:
            st.warning("The prepared test CSV contains no rows to score.")
            selected_features = None
        else:
            sample_index = int(st.number_input("Sample row", min_value=0, max_value=len(test_sample) - 1, value=0, step=1))
            selected_row = test_sample.iloc[sample_index]
            selected_features = {
                str(name): float(value)
                for name, value in selected_row.items()
                if pd.notna(value) and isinstance(value, Real) and not isinstance(value, bool)
            }
            api_default_features = {
                "merch_fraud_rate_30d",
                "merch_volume_30d",
                "txn_count_24h",
                "amt_sum_24h",
                "distinct_productcd_24h",
            }
            missing_defaults = sorted(api_default_features.difference(selected_features))
            if missing_defaults:
                st.info(
                    "This prepared dataset does not contain these rolling features; "
                    "the scorer will use its API defaults for them: "
                    + ", ".join(missing_defaults)
                )
            st.dataframe(selected_row.to_frame("value"), use_container_width=True)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"Could not load prepared test data: {exc}")
        selected_features = None

    with st.form("score_transaction"):
        st.write("The selected row is sent as named, preprocessed model features.")
        score_clicked = st.form_submit_button("Score selected row", disabled=selected_features is None)

    st.caption(f"Scorer: {scorer_url or '(not configured)'}")
    if score_clicked:
        if not scorer_url:
            st.error("Enter a scorer URL in the sidebar.")
        else:
            try:
                response = requests.post(
                    f"{scorer_url}/predict",
                    json={"features": selected_features},
                    timeout=30,
                )
                response.raise_for_status()
                result = response.json()
                fraud_prob = float(result["fraud_prob"])
                threshold = float(result["threshold"])
                if not math.isfinite(fraud_prob) or not 0.0 <= fraud_prob <= 1.0:
                    raise ValueError("Scorer returned an invalid fraud probability.")
                if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
                    raise ValueError("Scorer returned an invalid decision threshold.")

                metric_col1, metric_col2 = st.columns(2)
                metric_col1.metric("Fraud probability", f"{fraud_prob:.1%}")
                metric_col2.metric("Decision threshold", f"{threshold:.1%}")
                if result.get("flag"):
                    st.warning("Flagged for review")
                else:
                    st.success("Below the model threshold")
            except requests.RequestException as exc:
                st.error(f"Could not reach the scorer at {scorer_url}: {exc}")
            except (ValueError, KeyError, TypeError) as exc:
                st.error(f"Could not score this request: {exc}")

with drift_col:
    st.subheader("Drift leaderboard")
    st.caption(f"Input directory: {demo_dir}")
    missing_files = [path.name for path in (baseline_path, production_path, importance_path) if not path.exists()]
    if missing_files:
        st.info(
            "Drift input files are missing: "
            + ", ".join(missing_files)
            + ". Prepare them with demo/prepare_drift_windows.py."
        )
    else:
        try:
            baseline = pd.read_parquet(baseline_path)
            production = pd.read_parquet(production_path)
            importance = json.loads(importance_path.read_text(encoding="utf-8"))
            if baseline.empty or production.empty:
                raise ValueError("Baseline and production windows must both contain rows.")
            if not isinstance(importance, dict):
                raise ValueError("Importance JSON must contain a feature-to-weight object.")

            kl = kl_per_feature(baseline, production, bins=bins)
            if not kl:
                raise ValueError("The windows have no shared numeric features to compare.")
            ranked = cost_weighted_drift(kl, importance, p_fraud=p_fraud, c_fn=c_fn, c_fp=c_fp)
            chart_data = ranked.head(top_n).set_index("feature")["cost_drift_usd"]

            metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1.metric("Baseline rows", f"{len(baseline):,}")
            metric_col2.metric("Production rows", f"{len(production):,}")
            metric_col3.metric("Compared features", f"{len(kl):,}")
            st.bar_chart(chart_data, horizontal=True)
            st.dataframe(ranked.head(top_n), use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Could not calculate drift: {exc}")
