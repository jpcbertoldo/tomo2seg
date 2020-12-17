import time
from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, ClassVar

import tensorflow as tf

from tomo2seg.logger import logger
from tomo2seg.hosts import Host


@dataclass
class BaseArgs(ABC):

    host: Host
    script_name: str  # script or notebook's name

    runid: Optional[int]
    random_state_seed: Optional[int]

    def __post_init__(self):

        if self.runid is None:
            self.runid = int(time.time())
            logger.info(f"Using auto runid={self.runid}")

        if self.random_state_seed is None:
            self.random_state_seed = int(time.time()) % 1000
            logger.info(f"Using auto random_state_seed={self.random_state_seed}")


@dataclass
class TrainArgs(BaseArgs):

    # versions of the train script compatible with this class
    versions: ClassVar = (8,)

    class EarlyStopMode(Enum):
        no_early_stop = 0

    class BatchSizeMode(Enum):
        try_max_and_fail = 0
        try_max_and_reduce = 1

    # None: continue from the latest model
    # 1: continue from model.autosaved_model_path
    # 2: continue from model.autosaved2_model_path
    # continue_from_autosave: Optional[int] = None
    class TrainMode(Enum):
        from_scratch = 0
        continuation_from_autosaved_model = 1
        continuation_from_autosaved2_best_model = 2
        continuation_from_latest_model = 3

        @property
        def is_continuation(self) -> bool:
            return self in (
                TrainArgs.TrainMode.continuation_from_autosaved_model,
                TrainArgs.TrainMode.continuation_from_autosaved2_best_model,
                TrainArgs.TrainMode.continuation_from_latest_model,
            )

    early_stop_mode: EarlyStopMode
    batch_size_mode: BatchSizeMode
    train_mode: TrainMode

    volume_name: str
    volume_version: str
    labels_version: str

    partition_train: str = "train"
    partition_val: str = "val"

    model_fold: int = 0

    # override the auto-sized value
    # this allows to reproduce reproduce the same conditions across experiments
    batch_size_per_gpu: Optional[int] = None

    def __post_init__(self):

        super().__post_init__()

        if self.train_mode.is_continuation:
            assert self.runid is not None, f"Incompatible args {self.runid=} {self.self.train_mode=}"

        if self.batch_size_per_gpu is not None:
            assert self.batch_size_per_gpu > 0, f"{self.batch_size_per_gpu=}"

            ngpus = len(tf.config.list_physical_devices('GPU'))
