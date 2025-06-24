# app/model_loader.py
from __future__ import annotations

import logging, tempfile
from datetime import datetime, timezone
from typing import Any, Dict

import boto3, joblib

from .config import ml_model_config as cfg

log = logging.getLogger(__name__)
s3  = boto3.client("s3")


# ────────────────────────────────────────────────────────────────
# Internal helper: find newest artefact
# ────────────────────────────────────────────────────────────────
def _latest_s3_model(bucket: str, prefix: str, artefact: str) -> str:
    paginator = s3.get_paginator("list_objects_v2")
    newest_key, newest_time = None, datetime(1970, 1, 1, tzinfo=timezone.utc)

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(artefact) and obj["LastModified"] > newest_time:
                newest_key, newest_time = key, obj["LastModified"]

    if newest_key is None:
        raise FileNotFoundError("No matching model found in S3")
    return newest_key


# ────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────
def load_latest_model() -> Dict[str, Any]:
    """Download the latest model artefact from S3 and return the un-pickled object."""
    fb  = cfg.s3_fallback
    key = _latest_s3_model(fb.bucket, fb.root_prefix, fb.artifact_subpath)
    log.info("Downloading model artefact: s3://%s/%s", fb.bucket, key)

    with tempfile.NamedTemporaryFile("wb+", delete=False) as tmp:
        s3.download_fileobj(fb.bucket, key, tmp)
        tmp.flush()
        tmp.seek(0)
        model = joblib.load(tmp)

    return model
