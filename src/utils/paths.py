"""Central path definitions for the project."""
from pathlib import Path

ROOT        = Path(__file__).resolve().parents[2]
DATA_RAW    = ROOT / "data" / "raw"
DATA_PROC   = ROOT / "data" / "processed"
DATA_EXT    = ROOT / "data" / "external"
MODELS      = ROOT / "models" / "saved"
CHECKPOINTS = ROOT / "models" / "checkpoints"
OUTPUTS     = ROOT / "outputs"
LOGS        = ROOT / "logs"
CONFIGS     = ROOT / "configs"
DOCS        = ROOT / "docs"
