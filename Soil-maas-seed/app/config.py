# app/config.py  ← replaces your current file
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

# ────────────────────────────────────────────────────────────────
# 0)  Custom error to signal mis-configuration
# ────────────────────────────────────────────────────────────────
class ConfigError(RuntimeError):
    """Raised when a required configuration value is missing or invalid."""
    pass


# ────────────────────────────────────────────────────────────────
# 1)  Environment constants (validated)
# ────────────────────────────────────────────────────────────────
def _env(key: str, default: str | None = None) -> str:
    """Get env var or raise ConfigError if absent & no default."""
    val = os.getenv(key, default)
    if val in (None, ""):
        raise ConfigError(f"Environment variable {key} is not set")
    return val


MODEL_DIR: str = _env("MODEL_DIR", "./models")
REDIS_URL: str = _env("REDIS_URL", "redis://localhost:6379/0")
MODEL_CACHE_KEY: str = "latest_model"


# ────────────────────────────────────────────────────────────────
# 2)  Structured model-download settings
# ────────────────────────────────────────────────────────────────
@dataclass(slots=True, frozen=True)
class S3Fallback:
    bucket: str
    root_prefix: str
    artifact_subpaths: Tuple[str, ...]  # now a tuple of strings

    def __post_init__(self):
        if not self.bucket.strip():
            raise ConfigError("s3_fallback.bucket cannot be blank")

        if not self.root_prefix.endswith("/"):
            raise ConfigError("s3_fallback.root_prefix must end with '/'")

        if not self.artifact_subpaths:
            raise ConfigError("s3_fallback.artifact_subpaths must include at least one file")

        for subpath in self.artifact_subpaths:
            filename = Path(subpath).name
            if "." not in filename:
                raise ConfigError(
                    f"Invalid artifact_subpath '{subpath}': must include a file name with an extension"
                )



@dataclass(slots=True, frozen=True)
class ModelConfig:
    experiment_name: str
    artifact_path: str
    metric: str
    higher_is_better: bool
    s3_fallback: S3Fallback
    validate_metric_set: Tuple[str, ...] = field(
        default=("accuracy", "precision", "recall", "f1_score", "roc_auc")
    )

    def __post_init__(self):
        if not self.experiment_name.strip():
            raise ConfigError("experiment_name cannot be blank")

        # quick guard against fat-fingered MLflow artifact paths
        if "/" in self.artifact_path or " " in self.artifact_path:
            raise ConfigError("artifact_path must be a simple MLflow name, no slashes")

        if self.metric not in self.validate_metric_set:
            raise ConfigError(
                f"metric '{self.metric}' not recognised. "
                f"Choose from {self.validate_metric_set}"
            )


# ────────────────────────────────────────────────────────────────
# 3)  Concrete instance used by the app
# ────────────────────────────────────────────────────────────────
ml_model_config = ModelConfig(
    experiment_name="soil_experiment",
    artifact_path="soil_model",
    metric="f1_score",
    higher_is_better=True,
    s3_fallback=S3Fallback(
        bucket="agtech-models",
        root_prefix="Agtech-models-V1/",
        artifact_subpaths=("Soil_recomednation_weather_soil.pkl",
                          "lon_lat_weather_soil.pkl",
                          "label_encoder_weather_soil.pkl"
                          ),
    ),
)

# if __name__ == "__main__":
#     print("MODEL_DIR:", MODEL_DIR)
#     print("REDIS_URL:", REDIS_URL)
#     print("Model config loaded successfully:")
#     print(ml_model_config)
