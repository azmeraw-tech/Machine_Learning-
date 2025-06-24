# app/main.py
from __future__ import annotations

import logging
from fastapi import FastAPI
from . import api, model_loader, redis_cache

# ────────────────────────────────────────────────────────────────
# 1)  Logging
# ────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ────────────────────────────────────────────────────────────────
# 2)  Custom startup error
# ────────────────────────────────────────────────────────────────
class StartupError(RuntimeError):
    """Raised when the app cannot warm-up the model cache."""
    pass


app = FastAPI()


@app.on_event("startup")
async def startup_event() -> None:
    try:
        logger.info("Loading latest model artefact…")
        model = model_loader.load_latest_model()  # may raise error
        redis_cache.cache_model(model)            # may raise error
        logger.info("Model loaded & cached successfully.")

        # Test retrieval right after caching
        cached = redis_cache.get_cached_model()
        if cached is None:
            logger.error("Model was not cached properly!")
        else:
            logger.info("Model caching verified successfully.")

    except Exception as e:
        logger.critical("Startup failed: %s", e, exc_info=True)
        raise StartupError("Failed to initialise model cache") from e


# # ────────────────────────────────────────────────────────────────
# # 3)  Startup warm-up with exception handling
# # ────────────────────────────────────────────────────────────────
# @app.on_event("startup")
# async def startup_event() -> None:
#     try:
#         logger.info("Loading latest model artefact…")
#         model = model_loader.load_latest_model()                   # may raise ModelDownloadError
#         redis_cache.cache_model(model)                             # may raise RedisConnectionError
#         logger.info("Model loaded & cached successfully.")
#         print("sucessful")
#     except Exception as e:
#         # Log root cause and abort app start-up.
#         logger.critical("Startup failed: %s", e, exc_info=True)
#         # Raising any exception here forces Uvicorn to exit with non-zero,
#         # which is what Kubernetes / Docker health-checks expect.
#         raise StartupError("Failed to initialise model cache") from e


# ────────────────────────────────────────────────────────────────
# 4)  Business endpoints
# ────────────────────────────────────────────────────────────────
app.include_router(api.router)


