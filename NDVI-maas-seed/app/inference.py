# app/inference.py

from __future__ import annotations
import logging, os
from typing import Dict
from scipy.spatial import KDTree
import pandas as pd
import requests
from requests.exceptions import HTTPError, RequestException, Timeout

logger = logging.getLogger(__name__)

class FeatureFetchError(RuntimeError): ...
class InferenceError(RuntimeError):    ...

FEATURE_ORDER = ["longitude", "latitude"]

def find_closest_fico(kdtree: KDTree, ficos, lat: float, lon: float) -> float:
    _, index = kdtree.query([lat, lon])
    return float(ficos[index].round(2))


def run_inference(model: object, payload_from_client: Dict, kdtree: KDTree, ficos) -> Dict:
    """
    Fetch features from Feast, score with NDVI `kdtree`/`ficos`, return NDVI score.
    """
    FEAST_BASE_URL = os.getenv("FEAST_BASE_URL", "http://localhost:6567")

    # 1) Pull online features from Feast
    feast_req = {
        "features": [f"agri_tech_fv:{f}" for f in FEATURE_ORDER],
        "entities": {"customerId": [payload_from_client["customerId"]]},
    }

    try:
        resp = requests.post(f"{FEAST_BASE_URL}/get-online-features",
                             json=feast_req, timeout=3)
        resp.raise_for_status()
        meta, results = resp.json()["metadata"], resp.json()["results"]
    except (HTTPError, Timeout, RequestException) as e:
        logger.error("Feast request failed: %s", e, exc_info=True)
        raise FeatureFetchError(str(e)) from e
    except (KeyError, ValueError) as e:
        logger.error("Unexpected Feast payload: %s", e, exc_info=True)
        raise FeatureFetchError("Invalid payload from Feast") from e

    # 2) Create DataFrame
    try:
        values = [
            (v[0] if isinstance(v, list) else v)
            for res in results for v in res["values"]
        ]
        df = (
            pd.DataFrame([values], columns=meta["feature_names"])
            .drop(columns=["customerId"])
            .loc[:, FEATURE_ORDER]
        )
    except (ValueError, KeyError) as e:
        logger.error("Feature frame construction failed: %s", e, exc_info=True)
        raise FeatureFetchError("Feature frame construction failed") from e

    # 3) Extract coordinates and calculate NDVI-based score
    try:
        lon, lat = df["longitude"].iloc[0], df["latitude"].iloc[0]
        ndvi_score = find_closest_fico(kdtree, ficos, lat, lon)
    except Exception as e:
        logger.error("NDVI scoring failed: %s", e, exc_info=True)
        raise InferenceError("NDVI scoring failed") from e

    # 4) Response
    return {
        "ndvi_score": ndvi_score
    }



# # app/inference.py  – run_inference only
# from __future__ import annotations

# import logging, os
# from typing import Dict
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# import ast
# import pandas as pd
# import requests
# from requests.exceptions import HTTPError, RequestException, Timeout
# from scipy.special import expit                     # sigmoid

# logger = logging.getLogger(__name__)
# from scipy.spatial import KDTree

# class FeatureFetchError(RuntimeError): ...
# class InferenceError(RuntimeError):    ...

# # KEEP FEATURE_ORDER IN SYNC WITH TRAINING
# FEATURE_ORDER =["longitude",
#                 "latitude"]


# # ────────────────────────────────────────────────────────────────
# def run_inference(model: object, payload_from_client: Dict) -> Dict:
#     """Fetch features from Feast, score with `model`, return 300-850 credit score."""
#     FEAST_BASE_URL = os.getenv("FEAST_BASE_URL", "http://localhost:6567")

#     # 1) Pull online features
#     feast_req = {
#         "features": [f"agri_tech_fv:{f}" for f in FEATURE_ORDER],
#         "entities": {"customerId": [payload_from_client["customerId"]]},
#     }
#     try:
#         resp = requests.post(f"{FEAST_BASE_URL}/get-online-features",
#                              json=feast_req, timeout=3)
#         resp.raise_for_status()
#         meta, results = resp.json()["metadata"], resp.json()["results"]
#     except (HTTPError, Timeout, RequestException) as e:
#         logger.error("Feast request failed: %s", e, exc_info=True)
#         raise FeatureFetchError(str(e)) from e
#     except (KeyError, ValueError) as e:
#         logger.error("Unexpected Feast payload: %s", e, exc_info=True)
#         raise FeatureFetchError("Invalid payload from Feast") from e

#     # 2) DataFrame (training order) + type fixes
#     try:
#         values = [
#             (v[0] if isinstance(v, list) else v)
#             for res in results for v in res["values"]
#         ]
#         df = (
#             pd.DataFrame([values], columns=meta["feature_names"])
#             .drop(columns=["customerId"])
#             .loc[:, FEATURE_ORDER]
#         )
#     except (ValueError, KeyError) as e:
#         logger.error("Feature frame construction failed: %s", e, exc_info=True)
#         raise FeatureFetchError("Feature frame construction failed") from e

#     if "Proximity to Market" in df.columns:
#         df["Proximity to Market"] = df["Proximity to Market"].astype(int)

#     # 3) Predict
#     try:
#         raw_pred = model.predict(df)          # shape (1,) or (1,1)
#         prob_pos = float(expit(raw_pred)[0])  # sigmoid -> [0–1]
#         credit_score = int(round(300 + prob_pos * 550))
#     except Exception as e:
#         logger.error("Model prediction failed: %s", e, exc_info=True)
#         raise InferenceError("Model prediction failed") from e

#     # 4) Response (no feature importance)
#     return {
#         "credit_score": credit_score,
#         "probability_positive": prob_pos,
#     }












# def find_closest_fico(kdtree, ficos, lat, lon):
#     _, index = kdtree.query([lat, lon])
#     return ficos[index].round(2)

# # ────────────────────────────────────────────────────────────────
# def run_inference(model: object, payload_from_client: Dict) -> Dict:
#     """Fetch features from Feast, score with `model`, return 300-850 credit score."""
#     FEAST_BASE_URL = os.getenv("FEAST_BASE_URL", "http://localhost:6567")

#     # 1) Pull online features
#     feast_req = {
#         "features": [f"agri_tech_fv:{f}" for f in FEATURE_ORDER],
#         "entities": {"customerId": [payload_from_client["customerId"]]},
#     }
#     try:
#         resp = requests.post(f"{FEAST_BASE_URL}/get-online-features",
#                              json=feast_req, timeout=3)
#         resp.raise_for_status()
#         meta, results = resp.json()["metadata"], resp.json()["results"]
#     except (HTTPError, Timeout, RequestException) as e:
#         logger.error("Feast request failed: %s", e, exc_info=True)
#         raise FeatureFetchError(str(e)) from e
#     except (KeyError, ValueError) as e:
#         logger.error("Unexpected Feast payload: %s", e, exc_info=True)
#         raise FeatureFetchError("Invalid payload from Feast") from e

#     # 2) DataFrame (training order) + type fixes
#     try:
#         values = [
#             (v[0] if isinstance(v, list) else v)
#             for res in results for v in res["values"]
#         ]
#         df = (
#             pd.DataFrame([values], columns=meta["feature_names"])
#             .drop(columns=["customerId"])
#             .loc[:, FEATURE_ORDER]
#         )
#         lat = df["latitude"].values[0]
#         lon = df["longitude"].values[0]

#     except (ValueError, KeyError) as e:
#         logger.error("Feature frame construction failed: %s", e, exc_info=True)
#         raise FeatureFetchError("Feature frame construction failed") from e

#     # kdtree_data = load_kdtree(df)
#     # ndvi_score = find_closest_fico(df['kdtree'], df['ficos'], lat, lon)

#     kdtree = df['kdtree']
#     ficos = df['ficos']


#     # 3) Predict
#     try:
#         _, index = kdtree.query([lat, lon])
#         credit_score=ficos[index].round(2)
#     except Exception as e:
#         logger.error("Model prediction failed: %s", e, exc_info=True)
#         raise InferenceError("Model prediction failed") from e

#     # 4) Response (no feature importance)
#     return {
#         "credit_score": credit_score
#         # "probability_positive": prob_pos,
#     }











