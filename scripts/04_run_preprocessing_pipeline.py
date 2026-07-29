import argparse
import importlib.util
from pathlib import Path

from fraud_data_common import DEFAULT_OUTPUT_ROOT, add_common_args, ensure_output_dir

import os, sys

os.environ["HADOOP_HOME"] = r"C:\hadoop"
os.environ["hadoop.home.dir"] = r"C:\hadoop"
os.environ["PATH"] = r"C:\hadoop\bin;" + os.environ["PATH"]
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the local preprocessing audit pipeline described in the GCP plan: "
            "missing-value profiling, high-correlation pruning analysis, and weak "
            "label-correlation analysis."
        )
    )
    add_common_args(parser)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "preprocessing",
        help="Root directory for joined data, filtered data, and reports.",
    )
    parser.add_argument(
        "--drop-column-missing-threshold",
        type=float,
        default=95.0,
        help="Drop columns whose missing-value percentage is above this threshold.",
    )
    parser.add_argument(
        "--drop-row-missing-threshold",
        type=float,
        default=5.0,
        help="Drop rows with any missing values when the row-level missing rate is at or below this threshold.",
    )
    parser.add_argument(
        "--numeric-fill-floor",
        type=float,
        default=-999.0,
        help="Fallback numeric fill value; step 1 uses min(this value, feature_min - 1).",
    )
    parser.add_argument("--corr-threshold", type=float, default=0.95)
    parser.add_argument("--label-corr-threshold", type=float, default=0.01)
    return parser.parse_args()


def load_script_module(script_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    ensure_output_dir(args.output_root)

    scripts_dir = Path(__file__).resolve().parent
    step1 = load_script_module(scripts_dir / "01_check_missing_values.py", "step_missing")
    step2 = load_script_module(scripts_dir / "02_remove_correlated_features.py", "step_corr")
    step3 = load_script_module(
        scripts_dir / "03_remove_useless_label_corr_features.py", "step_label_corr"
    )

    step1_args = step1.parse_args.__globals__["argparse"].Namespace(
        transaction_csv=args.transaction_csv,
        identity_csv=args.identity_csv,
        join_how=args.join_how,
        sample_rows=args.sample_rows,
        spark_master=args.spark_master,
        spark_shuffle_partitions=args.spark_shuffle_partitions,
        spark_driver_memory=args.spark_driver_memory,
        output_csv=args.output_root / "01_joined_train.csv",
        reports_dir=args.output_root / "reports_missing",
        drop_column_missing_threshold=args.drop_column_missing_threshold,
        drop_row_missing_threshold=args.drop_row_missing_threshold,
        numeric_fill_floor=args.numeric_fill_floor,
    )
    step2_args = step2.parse_args.__globals__["argparse"].Namespace(
        transaction_csv=args.transaction_csv,
        identity_csv=args.identity_csv,
        join_how=args.join_how,
        sample_rows=args.sample_rows,
        spark_master=args.spark_master,
        spark_shuffle_partitions=args.spark_shuffle_partitions,
        spark_driver_memory=args.spark_driver_memory,
        output_csv=args.output_root / "02_no_high_corr.csv",
        reports_dir=args.output_root / "reports_feature_corr",
        corr_threshold=args.corr_threshold,
    )
    step3_args = step3.parse_args.__globals__["argparse"].Namespace(
        transaction_csv=args.transaction_csv,
        identity_csv=args.identity_csv,
        join_how=args.join_how,
        sample_rows=args.sample_rows,
        spark_master=args.spark_master,
        spark_shuffle_partitions=args.spark_shuffle_partitions,
        spark_driver_memory=args.spark_driver_memory,
        output_csv=args.output_root / "03_label_corr_filtered.csv",
        reports_dir=args.output_root / "reports_label_corr",
        label_corr_threshold=args.label_corr_threshold,
    )

    for module, module_args in (
        (step1, step1_args),
        (step2, step2_args),
        (step3, step3_args),
    ):
        original = module.parse_args
        module.parse_args = lambda module_args=module_args: module_args
        try:
            module.main()
        finally:
            module.parse_args = original

    print(f"Completed preprocessing pipeline under {args.output_root}")


if __name__ == "__main__":
    main()
