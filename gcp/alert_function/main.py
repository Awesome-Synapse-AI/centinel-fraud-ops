import math
from numbers import Real

import functions_framework


@functions_framework.http
def notify(request):
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return {"error": "Expected a JSON object."}, 400

    expected_loss = body.get("total_expected_loss_usd")
    top_feature = body.get("top_feature")
    top_feature_cost = body.get("top_feature_cost_usd")
    if isinstance(expected_loss, bool) or not isinstance(expected_loss, Real):
        return {"error": "total_expected_loss_usd must be numeric."}, 400
    if not math.isfinite(expected_loss) or expected_loss < 0:
        return {"error": "total_expected_loss_usd must be finite and non-negative."}, 400
    if not isinstance(top_feature, str) or not top_feature.strip():
        return {"error": "top_feature must be a non-empty string."}, 400
    if isinstance(top_feature_cost, bool) or not isinstance(top_feature_cost, Real):
        return {"error": "top_feature_cost_usd must be numeric."}, 400
    if not math.isfinite(top_feature_cost) or top_feature_cost < 0:
        return {"error": "top_feature_cost_usd must be finite and non-negative."}, 400

    return {
        "status": "accepted",
        "total_expected_loss_usd": float(expected_loss),
        "top_feature": top_feature.strip(),
        "top_feature_cost_usd": float(top_feature_cost),
    }, 200
