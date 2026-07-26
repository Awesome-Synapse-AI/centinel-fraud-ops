import functions_framework


@functions_framework.http
def notify(request):
    body = request.get_json(silent=True) or {}
    return {"status": "notified", "expected_loss_usd": body.get("total_expected_loss_usd", 0)}, 200
