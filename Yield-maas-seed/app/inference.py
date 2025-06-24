# app/inference.py  – run_inference only
from __future__ import annotations
from scipy.spatial import KDTree

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
FEATURE_ORDER =["region","crop_type","latitude", "longitude", "land_area", "yield_estimation_year"]


# Mappings and crop statistics
region_mapping = {'Afar': 0, 'Amhara': 1, 'Benushangul Gumuz': 2, 'DIRE DAWA': 3,
                  'Gambela': 4, 'Harari': 5, 'Oromiya': 6, 'S.N.N.P.R': 7,
                  'Somalia': 8, 'Tigray': 9}
crop_type_mapping = {'Barley': 0, 'Maize': 1, 'Sorghum': 3, 'Teff': 4, 'Wheat': 5, 'Potato': 2}
crop_yield_stats = {
    "Teff": {"min": 83.5, "max": 38766}, "Barley": {"min": 22.5, "max": 39984},
    "Wheat": {"min": 193.5, "max": 47096}, "Sorghum": {"min": 237, "max": 14882000},
    "Maize": {"min": 202, "max": 1608600}, "Potato": {"min": 336, "max": 193200}
}


crop_price_stats = {
    "Teff": {"min": 2.17, "max": 99.52}, "Barley": {"min": 0.89, "max": 37.37},
    "Wheat": {"min": 1.29, "max": 44.34}, "Sorghum": {"min": 0.9, "max": 34.78},
    "Maize": {"min": 0.57, "max": 29.32}, "Potato": {"min": 0.51, "max": 15.59}
}


def normalize_dict_keys(d: dict) -> dict:
    return {k.lower(): v for k, v in d.items()}

# Normalized versions (for case-insensitive lookup)
region_mapping = normalize_dict_keys(region_mapping)
crop_type_mapping = normalize_dict_keys(crop_type_mapping)
crop_yield_stats = normalize_dict_keys(crop_yield_stats)
crop_price_stats = normalize_dict_keys(crop_price_stats)

def map_value(mapping, value, label):
    key = value.lower()
    if key not in mapping:
        raise ValueError(f"{label} '{value}' is not in the mapping.")
    return mapping[key]


def normalize_yield(crop_type, yield_value):
    stats = crop_yield_stats.get(crop_type.lower())
    if not stats:
        raise ValueError(f"Crop type '{crop_type}' is not in the yield stats.")
    return (yield_value - stats["min"]) / (stats["max"] - stats["min"])

##----------------------------------------------------------------
def find_closest_fico(kdtree: KDTree, ficos, lat: float, lon: float) -> float:
    _, index = kdtree.query([lat, lon])
    return float(ficos[index].round(2))


# def calculate_fico_score(region, crop_type, lat, lon, land_area, year, model_path, ndvi_path):
#     # NDVI score
#     # kdtree_data = load_kdtree(ndvi_path)
#     ndvi_score = find_closest_fico(kdtree_data['kdtree'], kdtree_data['ficos'], lat, lon)



# # ────────────────────────────────────────────────────────────────
def run_inference(model: object,encoder:object,lonlat_mapper:object,Yield:object,payload_from_client: Dict, kdtree: KDTree, ficos) -> Dict:
    """Fetch features from Feast, score with `model`, return 300-850 credit score."""
    FEAST_BASE_URL = os.getenv("FEAST_BASE_URL", "http://localhost:6567")

    # 1) Pull online features
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
        lon = df["longitude"].iloc[0]
        lat = df["latitude"].iloc[0]
        cultivated_crop_type = df["crop_type"].iloc[0]
        region = df["region"].iloc[0]
        year = df["yield_estimation_year"].iloc[0]
        land_area = df["land_area"].iloc[0]
        

    except (ValueError, KeyError) as e:
        logger.error("Feature frame construction failed: %s", e, exc_info=True)
        raise FeatureFetchError("Feature frame construction failed") from e

    predicted_soil_features = {
        feature: mapper.predict(pd.DataFrame([[lon, lat]], columns=["lon", "lat"]))[0]
        for feature, mapper in lonlat_mapper.items()
    }

    training_features = model.feature_names_in_

    soil_input = pd.DataFrame({"lon": [lon], "lat": [lat], **predicted_soil_features})
    soil_input = soil_input[training_features]

#Yield
    region_encoded = map_value(region_mapping, region, "Region")
    crop_encoded = map_value(crop_type_mapping, cultivated_crop_type, "Crop type")
    yield_input = pd.DataFrame({"Region": [region_encoded], "crop type": [crop_encoded], "Year": [year]})

    predicted_yield = Yield.predict(yield_input)[0]
    normalized_yield = normalize_yield(cultivated_crop_type, predicted_yield)

#NDVI
    # find_closest_fico(kdtree_data['kdtree'], kdtree_data['ficos'], lat, lon)
    # 3) Predict
    try:
        lon, lat = df["longitude"].iloc[0], df["latitude"].iloc[0]
        ndvi_score = find_closest_fico(kdtree, ficos, lat, lon)
        predicted_crop_encoded = model.predict(soil_input)
        predicted_crop = encoder.inverse_transform(predicted_crop_encoded)[0]
        crop_similarity_matrix = pd.DataFrame(
            [
                [1.0, 0.2, 0.3, 0.2, 0.1, 0.2, 0.2, 0.5, 0.3, 0.2, 0.3, 0.4, 0.6, 0.4, 0.3, 0.2],  # Coffee
                [0.2, 1.0, 0.6, 0.7, 0.2, 0.6, 0.5, 0.2, 0.3, 0.8, 0.4, 0.5, 0.4, 0.5, 0.3, 0.2],  # Maize
                [0.3, 0.6, 1.0, 0.7, 0.3, 0.6, 0.5, 0.3, 0.3, 0.6, 0.4, 0.4, 0.3, 0.4, 0.3, 0.2],  # Grass
                [0.2, 0.7, 0.7, 1.0, 0.3, 0.7, 0.6, 0.2, 0.3, 0.7, 0.4, 0.4, 0.3, 0.4, 0.3, 0.2],  # Sorghum
                [0.1, 0.2, 0.3, 0.3, 1.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1],  # Fallow
                [0.2, 0.6, 0.6, 0.7, 0.2, 1.0, 0.7, 0.2, 0.3, 0.7, 0.4, 0.3, 0.3, 0.4, 0.3, 0.2],  # Teff
                [0.2, 0.5, 0.5, 0.6, 0.2, 0.7, 1.0, 0.2, 0.3, 0.8, 0.4, 0.3, 0.3, 0.4, 0.3, 0.2],  # Barley
                [0.5, 0.2, 0.3, 0.2, 0.2, 0.2, 0.2, 1.0, 0.3, 0.2, 0.3, 0.5, 0.6, 0.4, 0.3, 0.3],  # Enset
                [0.3, 0.3, 0.3, 0.3, 0.2, 0.3, 0.3, 0.3, 1.0, 0.6, 0.7, 0.4, 0.4, 0.5, 0.6, 0.3],  # Faba bean
                [0.2, 0.8, 0.6, 0.7, 0.2, 0.7, 0.8, 0.2, 0.6, 1.0, 0.4, 0.4, 0.3, 0.5, 0.3, 0.2],  # Wheat
                [0.3, 0.4, 0.4, 0.4, 0.2, 0.4, 0.4, 0.3, 0.7, 0.4, 1.0, 0.6, 0.4, 0.6, 0.5, 0.4],  # Haricot bean
                [0.4, 0.5, 0.4, 0.4, 0.1, 0.3, 0.3, 0.5, 0.4, 0.4, 0.6, 1.0, 0.7, 0.8, 0.5, 0.4],  # Vegetable
                [0.6, 0.4, 0.3, 0.3, 0.1, 0.3, 0.3, 0.6, 0.4, 0.3, 0.4, 0.7, 1.0, 0.6, 0.5, 0.4],  # Fruit
                [0.4, 0.5, 0.4, 0.4, 0.1, 0.4, 0.4, 0.3, 0.4, 0.4, 0.5, 0.7, 0.5, 1.0, 0.6, 0.5],  # Potato
                [0.3, 0.3, 0.3, 0.3, 0.1, 0.3, 0.3, 0.3, 0.6, 0.3, 0.5, 0.5, 0.5, 0.7, 1.0, 0.7],  # Pea
                [0.2, 0.2, 0.2, 0.2, 0.1, 0.2, 0.2, 0.3, 0.3, 0.2, 0.4, 0.4, 0.4, 0.6, 0.7, 1.0],  # Chat
            ],
            columns=[
                "Coffee", "Maize", "Grass", "Sorghum", "Fallow", "Teff", "Barley", "Enset",
                "Faba bean", "Wheat", "Haricot bean", "Vegetable", "Fruit", "Potato", "Pea", "Chat"
            ],
            index=[
                "Coffee", "Maize", "Grass", "Sorghum", "Fallow", "Teff", "Barley", "Enset",
                "Faba bean", "Wheat", "Haricot bean", "Vegetable", "Fruit", "Potato", "Pea", "Chat"
            ],
        )

        crop_similarity_matrix = crop_similarity_matrix.apply(lambda col: col.map(lambda x: round(x, 2)))
        cultivated_crop_type = cultivated_crop_type.lower()
        crop_similarity_lower = {k.lower(): {inner_k.lower(): v for inner_k, v in inner_dict.items()} for k, inner_dict in crop_similarity_matrix.items()}

        if cultivated_crop_type not in crop_similarity_lower:
            return "You asked for a product that is out of range."

        similarity_score = crop_similarity_lower.get(cultivated_crop_type, {}).get(predicted_crop.lower(), 0)
        credit_score = 300 + (similarity_score * 550)

        adjust_score = (credit_score + ndvi_score) / 2
        adjust_factor = adjust_score / 525
        fico_score = 300 + 550 * (normalized_yield * land_area * adjust_factor)
    except Exception as e:
        logger.error("Model prediction failed: %s", e, exc_info=True)
        raise InferenceError("Model prediction failed") from e

    # 4) Response (no feature importance)
    return {
        "credit_score": credit_score,
        "ndvi_score": ndvi_score,
        "fico_score": fico_score
    }


#   "credit_score": 575,
#   "ndvi_score": 734.24,










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


