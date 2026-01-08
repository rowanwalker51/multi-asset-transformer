from dataclasses import dataclass

@dataclass(frozen=False)
class TrainingConfig:
    batch_size: int
    lr: float
    epochs: int
    weight_decay: float
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    model_save_path: str