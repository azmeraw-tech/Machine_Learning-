
# --- imports are unchanged except for ClassVar + field_validator
from __future__ import annotations

import asyncio, logging
from typing import Any, Dict, ClassVar

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator  # <-- use v2-style validator

from .redis_cache import get_cached_model
from .inference import run_inference

router = APIRouter()
RETRY_LIMIT = 3

class APIRequestData(BaseModel):
    customerId: str = Field(..., min_length=1)
    loan_type: str
    source_bank: str

    # Tell Pydantic “these aren’t model fields—leave them alone”
    _valid_loan_types: ClassVar[set[str]]   = {"agtech_safee"}
    _valid_source_banks: ClassVar[set[str]] = {
        "coop", "enat", "zamzam", "wegagen", "bunna", "amhara"
    }

    # v2-style validators
    @field_validator("loan_type")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
    def _check_loan_type(cls, v: str):
        v_low = v.lower()
        if v_low not in cls._valid_loan_types:
            raise ValueError(f"loan_type must be one of {cls._valid_loan_types}")
        return v_low

    @field_validator("source_bank")
    def _check_source_bank(cls, v: str):
        v_low = v.lower()
        if v_low not in cls._valid_source_banks:
            raise ValueError(f"source_bank must be one of {cls._valid_source_banks}")
        return v_low

# Soil_recomednation_weather_soil.pkl",
#                           "lon_lat_weather_soil.pkl",
#                           "label_encoder_weather_soil.pkl",
#                           "yield_prediction_model.pkl",
#                           "NDVI.pkl"
@router.post("/Price_Estimation")
async def predict(data: APIRequestData) -> Dict[str, Any]:
    # Load multiple required models explicitly
    model = get_cached_model("Soil_recomednation_weather_soil")
    encoder= get_cached_model("label_encoder_weather_soil")
    lonlat_mapper= get_cached_model("lon_lat_weather_soil")
    ndvi= get_cached_model("NDVI")
    Yield= get_cached_model("yield_prediction_model")
    price= get_cached_model("price_predictor")


    if not all([model, encoder, lonlat_mapper,ndvi,Yield]):
        raise HTTPException(status_code=503, detail="One or more models not loaded")

    loop = asyncio.get_running_loop()
    # Check structure of model
    try:
        kdtree = ndvi["kdtree"]
        ficos = ndvi["ficos"]
        loaded_models = price['regressor']
        factor_model = price['factor_predictor']
    except (TypeError, KeyError):
        raise HTTPException(status_code=500, detail="Model missing 'kdtree' or 'ficos'")

    for attempt in range(RETRY_LIMIT):
        try:
            # pass all necessary models and the request data
            result = await loop.run_in_executor(
                None,
                run_inference,
                model, encoder, lonlat_mapper,Yield,
                data.dict(),kdtree, ficos,loaded_models,factor_model
            )
            return {
                "credit_score": result["price_score"],
                # "ndvi_score": result["ndvi_score"],
                "customerId": data.customerId,
            }
        except Exception as e:
            logging.warning("Attempt %d failed: %s", attempt + 1, e)
            await asyncio.sleep(0.5 * (attempt + 1))

    raise HTTPException(status_code=500, detail="Inference failed")


