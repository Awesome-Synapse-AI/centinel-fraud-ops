from datetime import timedelta

from feast import Entity, FeatureService, FeatureView, Field
from feast.infra.offline_stores.file_source import FileSource
from feast.types import Float32, Int64


card = Entity(name="card1", join_keys=["card1"])

tx_source = FileSource(
    path="data/staged/v1/",
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

fraud_v1 = FeatureService(name="fraud_v1", features=[tx_fv, merch_fv])
