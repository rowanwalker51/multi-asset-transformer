from dataclasses import dataclass
from typing import Dict

@dataclass(frozen=True)
class ModelConfig:
    d_model: int
    n_head: int
    n_layers: int
    n_classes: int
    dropout: float

@dataclass(frozen=True)
class AblationConfig:
    use_regime_embedding: bool
    shuffle_regime: bool
    constant_regime: bool
    use_stock_embedding: bool
    use_cls_token: bool


ABLATIONS: Dict[str, dict] = {
    "baseline": {},

    "no_regime": {
        "use_regime_embedding": False,
    },

    "shuffle_regime": {
        "shuffle_regime": True,
    },

    "constant_regime": {
        "constant_regime": True,
    },

    "no_stock_embedding": {
        "use_stock_embedding": False,
    },

    "no_cls_token": {
        "use_cls_token": False,
    },
}