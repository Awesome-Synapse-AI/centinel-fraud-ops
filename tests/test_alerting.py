import pandas as pd

from monitor.alerting import send_alert


class _Collection:
    def __init__(self):
        self.records = []

    def add(self, record):
        self.records.append(record)


class _Db:
    def __init__(self):
        self.alerts = _Collection()

    def collection(self, name):
        assert name == "alerts"
        return self.alerts


def test_send_alert_persists_compact_ranked_payload_without_webhook():
    db = _Db()
    ranked = pd.DataFrame([{"feature": "card1", "cost_drift_usd": 5012.5}])

    payload = send_alert(ranked, 7000.0, db=db, alert_function_url="")

    assert payload == {
        "total_expected_loss_usd": 7000.0,
        "top_feature": "card1",
        "top_feature_cost_usd": 5012.5,
    }
    assert db.alerts.records[0]["top_feature"] == "card1"
