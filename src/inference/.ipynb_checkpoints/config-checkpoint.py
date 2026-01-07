from dataclasses import dataclass

@dataclass(frozen=True)
class InferenceConfig:
    batch_size: int