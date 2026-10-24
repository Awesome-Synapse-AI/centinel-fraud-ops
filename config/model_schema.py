"""Shared, dependency-light contract for Feast model inputs and preparation."""

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

CATEGORICAL_RAW_FEATURE_NAMES = tuple(
    name.removesuffix("__encoded")
    for name in MODEL_FEATURE_NAMES
    if name.endswith("__encoded")
)

INTEGER_MODEL_FEATURES = frozenset(
    {"txn_count_24h", "distinct_productcd_24h", "merch_volume_30d"}
)

ROLLING_FEATURE_NAMES = (
    "txn_count_24h",
    "amt_sum_24h",
    "distinct_productcd_24h",
    "merch_fraud_rate_30d",
    "merch_volume_30d",
)

# Raw columns kept through correlation pruning so the contract can always be
# assembled later, even when a column is sparse or weakly correlated by itself.
PREPROCESS_REQUIRED_COLUMNS = frozenset(
    {
        *CATEGORICAL_RAW_FEATURE_NAMES,
        *(name for name in MODEL_FEATURE_NAMES if name not in ROLLING_FEATURE_NAMES and not name.endswith("__encoded")),
        "card1", "TransactionID", "TransactionDT", "ts", "isFraud", "has_identity",
    }
)

MODEL_METADATA_COLUMNS = ("TransactionID", "card1", "ts", "TransactionDT", "isFraud")


def expected_feature_type(name: str) -> str:
    """Return the BigQuery-compatible scalar type expected for a model input."""
    return "INT64" if name in INTEGER_MODEL_FEATURES else "FLOAT64"
