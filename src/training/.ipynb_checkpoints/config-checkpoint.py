from dataclasses import dataclass

@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    lr: float
    epochs: int
    weight_decay: float