from dataclasses import dataclass

@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    lr: float
    epochs: int
    weight_decay: float
    train_start: str
    train_end: str
    test_start: str
    test_end: str