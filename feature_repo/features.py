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

# This schema is the model-input contract emitted by
# scripts/04_run_preprocessing_pipeline.py and ingest/build_features.py.
# Do not include isFraud, identifiers, the event timestamp, or the entity key:
# isFraud would leak the target and the other fields are supplied separately in
# the training entity spine. Keep this list aligned with the staged parquet
# whenever the preprocessing contract changes.
MODEL_FEATURE_NAMES = (
    "TransactionAmt", "card2", "card3", "card5", "addr1", "addr2", "dist1", "dist2",
    "C2", "C5", "C9", "C13", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D10",
    "D11", "D12", "D13", "D14", "D15", "V30", "V70", "V91", "V129", "V131",
    "V159", "V160", "V162", "V165", "V202", "V203", "V207", "V213", "V214", "V215",
    "V218", "V263", "V264", "V265", "V267", "V273", "V274", "V276", "V291", "V292",
    "V303", "V309", "V310", "V312", "V313", "V314", "V315", "V337", "id_02", "id_03",
    "id_10", "id_11", "id_13", "id_14", "id_17", "id_18", "id_19", "id_20", "id_32",
    "has_identity", "ProductCD__encoded", "card4__encoded", "card6__encoded",
    "P_emaildomain__encoded", "R_emaildomain__encoded", "M1__encoded", "M2__encoded",
    "M3__encoded", "M4__encoded", "M5__encoded", "M6__encoded", "M7__encoded",
    "M8__encoded", "M9__encoded", "id_12__encoded", "id_15__encoded", "id_16__encoded",
    "id_28__encoded", "id_29__encoded", "id_30__encoded", "id_31__encoded",
    "id_33__encoded", "id_34__encoded", "id_35__encoded", "id_36__encoded",
    "id_37__encoded", "id_38__encoded", "DeviceType__encoded", "DeviceInfo__encoded",
    "txn_count_24h", "amt_sum_24h", "distinct_productcd_24h", "merch_fraud_rate_30d",
    "merch_volume_30d",
)
INTEGER_MODEL_FEATURES = {
    "txn_count_24h",
    "distinct_productcd_24h",
    "merch_volume_30d",
}

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
