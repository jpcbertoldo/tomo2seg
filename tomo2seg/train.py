import time
from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Optional, Type

import tensorflow as tf
from tomo2seg.args import BaseArgs
from tomo2seg.logger import logger
from tomo2seg.model import Model as T2SModel


@dataclass
class Args(BaseArgs):

    # versions of the train script compatible with this class
    versions: ClassVar = (8, 9)

    class EarlyStopMode(Enum):
        no_early_stop = 0

    class BatchSizeMode(Enum):
        try_max_and_fail = 0
        try_max_and_reduce = 1

    class TrainMode(Enum):
        from_scratch = 0
        continuation_from_autosaved_model = 1
        continuation_from_autosaved2_best_model = 2
        continuation_from_latest_model = 3

        @property
        def is_continuation(self) -> bool:
            return self in (
                self.continuation_from_autosaved_model,
                self.continuation_from_autosaved2_best_model,
                self.continuation_from_latest_model,
            )

    volume_name: str
    volume_version: str
    labels_version: str

    partition_train: str = "train"
    partition_val: str = "val"

    early_stop_mode: EarlyStopMode = EarlyStopMode.no_early_stop
    batch_size_mode: BatchSizeMode = BatchSizeMode.try_max_and_reduce
    train_mode: TrainMode = TrainMode.from_scratch

    model_fold: int = 0

    # override the auto-sized value
    # this allows to reproduce reproduce the same conditions across experiments
    batch_size_per_gpu: Optional[int] = None

    def __post_init__(self):

        super().__post_init__()

        if self.train_mode.is_continuation:
            assert (
                self.runid is not None
            ), f"Incompatible args {self.runid=} {self.self.train_mode=}"

        if self.batch_size_per_gpu is not None:
            assert self.batch_size_per_gpu > 0, f"{self.batch_size_per_gpu=}"


class TrainingFinished(Exception):
    pass


class FailedToFindBatchSize(Exception):
    pass
