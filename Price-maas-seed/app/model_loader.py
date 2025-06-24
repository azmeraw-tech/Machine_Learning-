# app/model_loader.py
from __future__ import annotations

import logging, tempfile
from datetime import datetime, timezone
from typing import Any, Dict

import boto3, joblib, cloudpickle
from pathlib import Path

from .config import ml_model_config as cfg

log = logging.getLogger(__name__)
s3 = boto3.client("s3")


def _latest_s3_model(bucket: str, prefix: str, artifact: str) -> str:
    paginator = s3.get_paginator("list_objects_v2")
    newest_key, newest_time = None, datetime(1970, 1, 1, tzinfo=timezone.utc)

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(artifact) and obj["LastModified"] > newest_time:
                newest_key, newest_time = key, obj["LastModified"]

    if newest_key is None:
        raise FileNotFoundError(f"No matching model '{artifact}' found in S3")
    return newest_key


def load_latest_model() -> Dict[str, Any]:
    """Download all specified model artefacts from S3 and return a dict of unpickled objects."""
    fb = cfg.s3_fallback
    loaded_models = {}

    for artifact_subpath in fb.artifact_subpaths:
        key = _latest_s3_model(fb.bucket, fb.root_prefix, artifact_subpath)
        log.info("Downloading model artefact: s3://%s/%s", fb.bucket, key)

        with tempfile.NamedTemporaryFile("wb+", delete=False) as tmp:
            s3.download_fileobj(fb.bucket, key, tmp)
            tmp.flush()
            tmp.seek(0)

            # Try joblib first, fallback to cloudpickle
            try:
                model = joblib.load(tmp)
            except Exception as e:
                log.warning(f"joblib failed to load model {artifact_subpath}: {e}")
                tmp.seek(0)
                model = cloudpickle.load(tmp)

        model_key = Path(artifact_subpath).stem
        loaded_models[model_key] = model

    return loaded_models





# # app/model_loader.py
# from __future__ import annotations

# import logging, tempfile
# from datetime import datetime, timezone
# from typing import Any, Dict

# import boto3, joblib

# from .config import ml_model_config as cfg

# log = logging.getLogger(__name__)
# s3  = boto3.client("s3")


# # ────────────────────────────────────────────────────────────────
# # Internal helper: find newest artefact
# # ────────────────────────────────────────────────────────────────
# def _latest_s3_model(bucket: str, prefix: str, artefact: str) -> str:
#     paginator = s3.get_paginator("list_objects_v2")
#     newest_key, newest_time = None, datetime(1970, 1, 1, tzinfo=timezone.utc)

#     for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
#         for obj in page.get("Contents", []):
#             key = obj["Key"]
#             if key.endswith(artefact) and obj["LastModified"] > newest_time:
#                 newest_key, newest_time = key, obj["LastModified"]

#     if newest_key is None:
#         raise FileNotFoundError("No matching model found in S3")
#     return newest_key


# # ────────────────────────────────────────────────────────────────
# # Public API
# # ────────────────────────────────────────────────────────────────
# def load_latest_model() -> Dict[str, Any]:
#     """Download the latest model artefact from S3 and return the un-pickled object."""
#     fb  = cfg.s3_fallback
#     key = _latest_s3_model(fb.bucket, fb.root_prefix, fb.artifact_subpath)
#     log.info("Downloading model artefact: s3://%s/%s", fb.bucket, key)

#     with tempfile.NamedTemporaryFile("wb+", delete=False) as tmp:
#         s3.download_fileobj(fb.bucket, key, tmp)
#         tmp.flush()
#         tmp.seek(0)
#         model = joblib.load(tmp)

#     return model
