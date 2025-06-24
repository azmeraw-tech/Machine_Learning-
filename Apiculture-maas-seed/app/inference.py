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
FEATURE_ORDER =[
            "Beeswax Production",
            "Mortality Rate (%)",
            "NDVI",
            "Production Period (months)",
            "Production Practice (Per Year)",
            "Training",
            "Methods to Control Predators and Pests",
            "Years of Experience",
            "Access to Domain Expert",
            "Proximity to Clean Water (km)",
            "Diversity of Flowering Plants",
            "Temperature (°C)",
            "Humidity (%)",
            "Honey Quality",
            "Honey Color",
            "Honey Harvesting Method",
            "Proximity to Market (km)",
            "Price per kg (ETB)",
            "Market Channels",
            "Hives Information",
            "Number of Hives",
            "Honey Bee Types",
            "Honey Yield per year (kg)"]

# ────────────────────────────────────────────────────────────────
def run_inference(model: object, payload_from_client: Dict) -> Dict:
    """Fetch features from Feast, score with `model`, return 300-850 credit score."""
    FEAST_BASE_URL = os.getenv("FEAST_BASE_URL", "http://localhost:6567")

    # 1) Pull online features
    feast_req = {
        "features": [f"apiculture_fv:{f}" for f in FEATURE_ORDER],
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
    
    if isinstance(df.get("Hives Information"), str):
        try:
            parsed_hives_info = ast.literal_eval(df["Hives Information"])
            df["Hives Information"] = parsed_hives_info
        except (SyntaxError, ValueError):
            df["Hives Information"] = {}
    # ---------------------------------------------------------
    # 2. Extract "Traditional", "Modern", "Transitional"
    # ---------------------------------------------------------
    for hive_type in ["Traditional", "Modern", "Transitional"]:
        df[hive_type] = df.get("Hives Information", {}).get(hive_type, 0)
    # ---------------------------------------------------------
    # 3. Convert "Hives Information" back to string if needed
    # ---------------------------------------------------------
    df["Hives Information"] = str(df.get("Hives Information", {}))
    # Convert "Honey Bee Types" to a single string if it's a list
    if isinstance(df.get("Honey Bee Types"), list):
        df["Honey Bee Types"] = ",".join(df["Honey Bee Types"])
    # Convert input data to a single-row DataFrame
    # df = pd.DataFrame([df])

    if df.isnull().all().all():
        raise ValueError("Borrow ID is not found or No agtech data for the specific borrower")

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





# def get_agtech_apiculture_model_predictor(input_data, apiculture_s3_model_url):
#     """
#     Predicts an Apiculture Score using a linear regression model.
#     Args:
#         input_data (dict): Input features for prediction.
#         apiculture_s3_model_url (str): S3 URL where the model is stored.
#     Returns:
#         dict: Predicted score and feature importance.
#     """
#     # Load the model from S3
#     api_model = read_models_s3(apiculture_s3_model_url)
#     # ---------------------------------------------------------
#     # 1. Convert "Hives Information" string to a dictionary
#     # ---------------------------------------------------------
#     if isinstance(input_data.get("Hives Information"), str):
#         try:
#             parsed_hives_info = ast.literal_eval(input_data["Hives Information"])
#             input_data["Hives Information"] = parsed_hives_info
#         except (SyntaxError, ValueError):
#             input_data["Hives Information"] = {}
#     # ---------------------------------------------------------
#     # 2. Extract "Traditional", "Modern", "Transitional"
#     # ---------------------------------------------------------
#     for hive_type in ["Traditional", "Modern", "Transitional"]:
#         input_data[hive_type] = input_data.get("Hives Information", {}).get(hive_type, 0)
#     # ---------------------------------------------------------
#     # 3. Convert "Hives Information" back to string if needed
#     # ---------------------------------------------------------
#     input_data["Hives Information"] = str(input_data.get("Hives Information", {}))
#     # Convert "Honey Bee Types" to a single string if it's a list
#     if isinstance(input_data.get("Honey Bee Types"), list):
#         input_data["Honey Bee Types"] = ",".join(input_data["Honey Bee Types"])
#     # Convert input data to a single-row DataFrame
#     input_df = pd.DataFrame([input_data])
#     # ---------------------------------------------------------
#     # 4. Check for all-missing input DataFrame
#     # ---------------------------------------------------------
#     if input_df.isnull().all().all():
#         raise ValueError("Borrow ID is not found or No agtech data for the specific borrower")
#     # Apply the model pipeline to process input and get predictions
#     raw_prediction = api_model.predict(input_df)
#     # Scale score between 300 and 850
#     scaled_score = round(300 + (raw_prediction[0] * 550), 2)
#     # Extract feature importance (for linear regression)
#     if hasattr(api_model.named_steps["model"], "coef_"):
#         feature_importance = api_model.named_steps["model"].coef_.flatten()  # Ensure 1D
#         feature_importance_dict = {
#             feature: weight for feature, weight in zip(input_df.columns, feature_importance)
#         }
#         # Get top 5 most important features (by absolute value)
#         top_5_features = [
#             {feature: weight}
#             for feature, weight in sorted(
#                 feature_importance_dict.items(),
#                 key=lambda item: abs(item[1]),
#                 reverse=True
#             )[:5]
#         ]
#     else:
#         top_5_features = []
#     # Construct the response
#     predictor_result = {
#         "score": float(scaled_score),
#         "feature_importance": top_5_features
#     }
#     return predictor_result

# # def run_inference(model: object, payload_from_client: Dict) -> Dict:
# #     """Fetch features from Feast, preprocess, score with `model`, return 300–850 credit score."""
# #     FEAST_BASE_URL = os.getenv("FEAST_BASE_URL", "http://localhost:6567")

# #     # 1) Pull online features from Feast
# #     feast_req = {
# #         "features": [f"apiculture_fv:{f}" for f in FEATURE_ORDER],
# #         "entities": {"customerId": [payload_from_client["customerId"]]},
# #     }
# #     try:
# #         resp = requests.post(f"{FEAST_BASE_URL}/get-online-features",
# #                              json=feast_req, timeout=3)
# #         resp.raise_for_status()
# #         meta, results = resp.json()["metadata"], resp.json()["results"]
# #     except (HTTPError, Timeout, RequestException) as e:
# #         logger.error("Feast request failed: %s", e, exc_info=True)
# #         raise FeatureFetchError(str(e)) from e
# #     except (KeyError, ValueError) as e:
# #         logger.error("Unexpected Feast payload: %s", e, exc_info=True)
# #         raise FeatureFetchError("Invalid payload from Feast") from e

# #     # 2) Convert to DataFrame in FEATURE_ORDER
# #     try:
# #         values = [
# #             (v[0] if isinstance(v, list) else v)
# #             for res in results for v in res["values"]
# #         ]
# #         df = (
# #             pd.DataFrame([values], columns=meta["feature_names"])
# #             .drop(columns=["customerId"], errors="ignore")
# #             .loc[:, FEATURE_ORDER]
# #         )
# #     except (ValueError, KeyError) as e:
# #         logger.error("Feature frame construction failed: %s", e, exc_info=True)
# #         raise FeatureFetchError("Feature frame construction failed") from e

# #     # 3) Preprocessing
# #     try:
# #         # Convert categorical columns
# #         categorical_columns = df.select_dtypes(include=['object', 'category']).columns
# #         label_encoder = LabelEncoder()
# #         for col in categorical_columns:
# #             try:
# #                 df[col] = label_encoder.fit_transform(df[col])
# #             except Exception:
# #                 df[col] = 0  # fallback for unseen/invalid values

# #         # Cast specific known features if needed
# #         if "Proximity to Market" in df.columns:
# #             df["Proximity to Market"] = df["Proximity to Market"].astype(int)

# #         # Optional: handle nested features like "Hives Information"
# #         # if "Hives Information" in df.columns:
# #         #     try:
# #         #         hives_info = ast.literal_eval(df["Hives Information"].iloc[0])
# #         #     except (ValueError, SyntaxError):
# #         #         hives_info = {}
# #         #     for hive_type in ["Traditional", "Modern", "Transitional"]:
# #         #         df[hive_type] = hives_info.get(hive_type, 0)
# #         #     df = df.drop(columns=["Hives Information"], errors="ignore")

# #         # Scale features
# #         scaler = StandardScaler()
# #         df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)

# #     except Exception as e:
# #         logger.error("Preprocessing failed: %s", e, exc_info=True)
# #         raise InferenceError("Feature preprocessing failed") from e

# #     # 4) Predict
# #     try:
# #         raw_pred = model.predict(df_scaled)          # shape (1,) or (1,1)
# #         prob_pos = float(expit(raw_pred[0]))         # sigmoid -> [0–1]
# #         credit_score = int(round(300 + prob_pos * 550))
# #     except Exception as e:
# #         logger.error("Model prediction failed: %s", e, exc_info=True)
# #         raise InferenceError("Model prediction failed") from e

# #     # 5) Response
# #     return {
# #         "credit_score": credit_score,
# #         "probability_positive": prob_pos,
# #     }



