"""
log_model_metrics.py
————————————
1. Download latest model artefact from S3
2. Log model + metrics to MLflow
3. Persist the metrics JSON back to S3
"""

from __future__ import annotations
import argparse, json, logging, os, tempfile
from pathlib import Path
from typing import Dict

import boto3, botocore, mlflow

# ────────────────────────  logging  ────────────────────────────
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# ───────────────  tiny domain-error helpers  ───────────────────
class S3DownloadError(RuntimeError): ...
class MLflowLogError(RuntimeError): ...
class S3UploadError(RuntimeError): ...

# ───────────────────────  config defaults  ─────────────────────
S3_URI = os.getenv(
    "MODEL_S3_URI",
    "s3://cs-infernece-model-testdata/models/models/1/Logistic_Regression_calibrated_model.joblib",
)
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "ml_test_v6")
METRICS: Dict[str, float] = {
    "f1_score": float(os.getenv("F1_SCORE", 0.84)),
    "accuracy": float(os.getenv("ACCURACY", 0.89)),
}
METRICS_S3_KEY = os.getenv("METRICS_S3_KEY", "models/models/1/model_metrics.json")

# ────────────────────────  utilities  ──────────────────────────
def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError("MODEL_S3_URI must start with s3://")
    bucket, key = uri[5:].split("/", 1)
    return bucket, key

def download_s3_file(s3, bucket: str, key: str) -> Path:
    dst = Path(tempfile.gettempdir()) / Path(key).name
    try:
        dst.parent.mkdir(exist_ok=True)
        s3.download_file(bucket, key, str(dst))
        logger.info("Downloaded model to %s", dst)
        return dst
    except botocore.exceptions.BotoCoreError as e:
        logger.error("S3 download failed: %s", e, exc_info=True)
        raise S3DownloadError(str(e)) from e

def log_to_mlflow(local_model: Path, metrics: Dict[str, float], uri: str) -> None:
    try:
        # Turn relative paths into URI expected by MLflow
        if not uri.startswith(("http://", "https://", "file://", "sqlite://")):
            uri = f"file://{Path(uri).resolve()}"
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run():
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(local_model), artifact_path="txn_model_named")
        logger.info("Logged to MLflow experiment '%s' at %s", MLFLOW_EXPERIMENT, uri)
    except Exception as e:
        logger.error("MLflow logging failed: %s", e, exc_info=True)
        raise MLflowLogError(str(e)) from e

def upload_metrics_json(s3, metrics: Dict[str, float], bucket: str, dst_key: str) -> None:
    tmp = Path(tempfile.gettempdir()) / "model_metrics.json"
    tmp.write_text(json.dumps(metrics))
    try:
        s3.upload_file(str(tmp), bucket, dst_key)
        logger.info("Uploaded metrics JSON to s3://%s/%s", bucket, dst_key)
    except botocore.exceptions.BotoCoreError as e:
        logger.error("S3 upload failed: %s", e, exc_info=True)
        raise S3UploadError(str(e)) from e

# ─────────────────────────  main  ──────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "./mlruns"),
        help="MLflow tracking URI (HTTP, DB, or local dir). "
             "Defaults to env var or ./mlruns.",
    )
    args = parser.parse_args()

    bucket, key = parse_s3_uri(S3_URI)
    s3 = boto3.client("s3")

    model_path = download_s3_file(s3, bucket, key)
    log_to_mlflow(model_path, METRICS, args.tracking_uri)
    upload_metrics_json(s3, METRICS, bucket, METRICS_S3_KEY)

if __name__ == "__main__":
    main()
