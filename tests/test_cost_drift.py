import pandas as pd

from monitor.cost_drift import cost_weighted_drift, kl_per_feature


def test_cost_weighted_drift_returns_ranked_frame():
    baseline = pd.DataFrame({"a": [0, 1, 2, 3], "b": [1, 1, 1, 1]})
    prod = pd.DataFrame({"a": [2, 3, 4, 5], "b": [1, 1, 1, 1]})
    kl = kl_per_feature(baseline, prod, bins=4)
    ranked = cost_weighted_drift(kl, {"a": 2.0, "b": 1.0}, p_fraud=0.02)
    assert list(ranked.columns) == ["feature", "kl", "importance", "cost_drift_usd", "cum_cost_drift"]
    assert ranked.iloc[0]["feature"] == "a"
