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


@router.post("/seed")
async def predict(data: APIRequestData) -> Dict[str, Any]:
    model = get_cached_model()
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    loop = asyncio.get_running_loop()

    for attempt in range(RETRY_LIMIT):
        try:
            result = await loop.run_in_executor(None, run_inference, model, data.dict())
            return {
                "credit_score": result["credit_score"],
                "customerId": data.customerId,
            }
        except Exception as e:
            logging.warning("Attempt %d failed: %s", attempt + 1, e)
            await asyncio.sleep(0.5 * (attempt + 1))

    raise HTTPException(status_code=500, detail="Inference failed")


