import functions_framework


@functions_framework.http
def notify(request):
    body = request.get_json(silent=True) or {}
    if not isinstance(body, dict):
        return {"error": "Expected a JSON object."}, 400
    expected_loss = body.get("total_expected_loss_usd", 0)
    try:
        expected_loss = float(expected_loss)
    except (TypeError, ValueError):
        return {"error": "total_expected_loss_usd must be numeric."}, 400
    return {"status": "notified", "expected_loss_usd": expected_loss}, 200
