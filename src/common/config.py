from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass(frozen=True)
class CommonConfig:
    num_stocks: int
    n_regimes: int
    seq_len: int
    random_seed: int


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def get_config_path(filename: str) -> Path:
    """Return absolute path to a config file."""
    return PROJECT_ROOT / "configs" / filename


def load_yaml(path: str | Path) -> dict:
    """
    Load a YAML file and return it as a dictionary.

    Args:
        path: Path to the YAML file.

    Returns:
        dict: contents of the YAML file
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"YAML config file not found: {path}")

    with path.open("r") as f:
        return yaml.safe_load(f)