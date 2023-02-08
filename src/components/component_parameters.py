from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ExampleGenParameters:
    should_use_local_sample_data: bool
    limit_dataset_size: Optional[int]


@dataclass(frozen=True)
class TrainerParameters:
    train_num_steps: int
    eval_num_steps: int
    epochs: int


@dataclass(frozen=True)
class EvaluatorParameters:
    metric_name: str
    metric_threshold: float


@dataclass(frozen=True)
class PusherParameters:
    serving_model_dir: str
