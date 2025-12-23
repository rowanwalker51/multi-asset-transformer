from dataclasses import dataclass

@dataclass(frozen=True)
class ModelConfig:
    d_model: int
    n_head: int
    n_layers: int
    n_classes: int
    dropout: float