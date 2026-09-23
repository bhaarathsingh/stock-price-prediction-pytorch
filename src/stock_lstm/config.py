from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class TrainConfig:
    lookback: int = 29
    hidden_size: int = 32
    num_layers: int = 2
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 0.001
    patience: int = 20
    train_fraction: float = 0.7
    validation_fraction: float = 0.1
    seed: int = 42
    device: str = "cpu"

    def __post_init__(self):
        for name in ("lookback", "hidden_size", "num_layers", "epochs", "batch_size", "patience"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not (0 < self.train_fraction < 1 and 0 < self.validation_fraction < 1):
            raise ValueError("Split fractions must each be between zero and one")
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("Split fractions must leave a test set")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("device must be cpu, cuda, or auto")
