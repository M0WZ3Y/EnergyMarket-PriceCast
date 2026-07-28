from pathlib import Path

import yaml

from src.models.base import BaseModel
from src.models.naive import NaiveModel
from src.models.sarimax import SARIMAXModel
from src.models.lear_lasso import LEARLassoModel

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "models.yaml"

__all__ = [
    "BaseModel",
    "NaiveModel",
    "SARIMAXModel",
    "LEARLassoModel",
    "load_models_config",
]


def load_models_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
