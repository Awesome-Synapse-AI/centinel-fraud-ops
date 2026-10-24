from datetime import timedelta
import os
import sys
from pathlib import Path

from feast import BigQuerySource, Entity, FeatureService, FeatureView, Field
from feast import ValueType
from feast.types import Float32, Float64, Int64

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.env import load_repo_env
from config.model_schema import INTEGER_MODEL_FEATURES, MODEL_FEATURE_NAMES


load_repo_env()


card = Entity(name="card1", join_keys=["card1"], value_type=ValueType.INT64)

tx_source = BigQuerySource(
    table=os.environ.get("FEAST_BIGQUERY_TABLE", "your-project.fraud_features.features_v1"),
    timestamp_field="ts",
)

tx_fv = FeatureView(
    name="tx_velocity",
    entities=[card],
    ttl=timedelta(days=30),
    schema=[
        Field(name="txn_count_24h", dtype=Float32),
        Field(name="amt_sum_24h", dtype=Float32),
        Field(name="distinct_productcd_24h", dtype=Int64),
    ],
    online=True,
    source=tx_source,
    tags={"layer": "velocity"},
)

merch_fv = FeatureView(
    name="merchant_risk",
    entities=[card],
    ttl=timedelta(days=90),
    schema=[
        Field(name="merch_fraud_rate_30d", dtype=Float32),
        Field(name="merch_volume_30d", dtype=Int64),
    ],
    online=True,
    source=tx_source,
    tags={"layer": "aggregate"},
)

model_input_fv = FeatureView(
    name="fraud_model_input",
    entities=[card],
    ttl=timedelta(days=90),
    schema=[
        Field(name=name, dtype=Int64 if name in INTEGER_MODEL_FEATURES else Float64)
        for name in MODEL_FEATURE_NAMES
    ],
    online=True,
    source=tx_source,
    tags={"layer": "model_input", "preprocessing_contract": "v1"},
)

fraud_v1 = FeatureService(name="fraud_v1", features=[model_input_fv])
