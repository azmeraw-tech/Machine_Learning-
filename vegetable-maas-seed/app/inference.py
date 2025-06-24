# app/inference.py  – run_inference only
from __future__ import annotations

import logging, os
from typing import Dict
from sklearn.preprocessing import LabelEncoder, StandardScaler
import ast
import pandas as pd
import requests
from requests.exceptions import HTTPError, RequestException, Timeout
from scipy.special import expit                     # sigmoid

logger = logging.getLogger(__name__)

class FeatureFetchError(RuntimeError): ...
class InferenceError(RuntimeError):    ...

# KEEP FEATURE_ORDER IN SYNC WITH TRAINING
FEATURE_ORDER =["land_area",
                "vegetable_type",
                "NDVI",
                "soil_fertility",
                "agriculture_practice",
                "harvest_frequency",
                "price_per_Kg",
                "yield_kg_per_hec",
                "proximity_to_market",
                "production_period",
                "market_channel"]


# # ────────────────────────────────────────────────────────────────
def run_inference(model: object, payload_from_client: Dict) -> Dict:
    """Fetch features from Feast, score with `model`, return 300-850 credit score."""
    FEAST_BASE_URL = os.getenv("FEAST_BASE_URL", "http://localhost:6567")

    # 1) Pull online features
    feast_req = {
        "features": [f"vegetable_production_fv:{f}" for f in FEATURE_ORDER],
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

    # 2) DataFrame (training order) + type fixes
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

    if df.isnull().all().all():
        raise ValueError("Borrow ID is not found or No agtech data for the specific borrower")
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    label_encoder = LabelEncoder()
    for col in categorical_columns:
        df[col] = label_encoder.fit_transform(df[col])
    scaler = StandardScaler()
    df = scaler.fit_transform(df)

    # 3) Predict
    try:
        raw_pred = model.predict(df)          # shape (1,) or (1,1)
        prob_pos = float(expit(raw_pred)[0])  # sigmoid -> [0–1]
        credit_score = int(round(300 + prob_pos * 550))
    except Exception as e:
        logger.error("Model prediction failed: %s", e, exc_info=True)
        raise InferenceError("Model prediction failed") from e

    # 4) Response (no feature importance)
    return {
        "credit_score": credit_score,
        "probability_positive": prob_pos,
    }













# def get_agtech_sectors_model_predictor(agtech_sector_model_url, new_data_df):
#     # ---------------------------------------------------------
#     # 0) Check for all-missing input DataFrame
#     # ---------------------------------------------------------
#     if new_data_df.isnull().all().all():
#         raise ValueError("Borrow ID is not found or No agtech data for the specific borrower")
#     print(new_data_df.info())
#     model = read_models_s3(agtech_sector_model_url)
#     categorical_columns = new_data_df.select_dtypes(include=['object', 'category']).columns
#     label_encoder = LabelEncoder()
#     for col in categorical_columns:
#         new_data_df[col] = label_encoder.fit_transform(new_data_df[col])
#     scaler = StandardScaler()
#     X_scaled_new = scaler.fit_transform(new_data_df)
#     predicted_score = model.predict(X_scaled_new)
#     final_score = round(300 + predicted_score[0] * 550, 2)
#     class_labels = np.round(predicted_score)
#     predict_proba = None  # Linear regression does not provide probability estimates
#     if hasattr(model, 'coef_'):
#         feature_importance = model.coef_
#         feature_importance_dict = {
#             feature: weight for feature, weight in zip(new_data_df.columns, feature_importance)
#         }
#         # Sort features by absolute weight and select the top 5
#         sorted_features = sorted(feature_importance_dict.items(), key=lambda item: abs(item[1]), reverse=True)[:5]
#         top_5_features = [{feature: weight} for feature, weight in sorted_features]
#     else:
#         top_5_features = []
#     predictor_result = {
#         "score": float(final_score),
#         "class_label": float(class_labels[0]) if len(class_labels) == 1 else class_labels.tolist(),
#         "predict_proba": (
#             predict_proba[0].tolist()
#             if predict_proba is not None and len(predict_proba) == 1
#             else predict_proba.tolist() if predict_proba is not None
#             else None
#         ),
#         "feature_importance": top_5_features
#     }
#     return predictor_result



# # ────────────────────────────────────────────────────────────────
# def run_inference(model: object, payload_from_client: Dict) -> Dict:
#     """Fetch features from Feast, score with `model`, return 300-850 credit score."""
#     FEAST_BASE_URL = os.getenv("FEAST_BASE_URL", "http://localhost:6567")

#     # 1) Pull online features
#     feast_req = {
#         "features": [f"seed_production_fv:{f}" for f in FEATURE_ORDER],
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


