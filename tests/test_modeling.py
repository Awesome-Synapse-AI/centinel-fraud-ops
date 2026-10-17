import numpy as np
import pandas as pd

from train.modeling import evaluate_model, save_bundle


class _ProbabilityModel:
    def predict_proba(self, features):
        probabilities = np.linspace(0.1, 0.9, len(features))
        return np.column_stack([1.0 - probabilities, probabilities])


def test_evaluation_with_single_class_test_split_does_not_block_saving(tmp_path):
    model = _ProbabilityModel()
    x_val = pd.DataFrame({"feature": [0, 1, 2, 3]})
    y_val = pd.Series([0, 1, 0, 1])
    x_test = pd.DataFrame({"feature": [4, 5, 6]})
    y_test = pd.Series([0, 0, 0])

    metrics = evaluate_model(model, x_val, y_val, x_test, y_test)
    output = tmp_path / "model.joblib"
    save_bundle(output, model, ["feature"], metrics, {"feature": 1.0})

    assert metrics["roc_auc_test"] is None
    assert output.exists()
    assert output.with_suffix(".metrics.json").exists()
