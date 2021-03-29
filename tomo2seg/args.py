import time
from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Optional, Type

import tensorflow as tf
from tomo2seg.hosts import Host, get_host
from tomo2seg.logger import logger


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

        if self.host is None:
            self.host = get_host()
            logger.info(f"Using auto host={self.host.hostname=}")


@dataclass
class ProcessVolumeArgs(BaseArgs):

    # versions of the train script compatible with this class
    versions: ClassVar = (5,)

    class AggregationStrategy(Enum):
        """This identifies the strategy used to deal with overlapping probabilities."""

        average_probabilities = 0

    class CroppingStrategy(Enum):
        """how to pick crop the size"""

        maximum_size = 0
        maximum_size_reduced_overlap = 1

    # todo integrate this to the model object instead
    class ModelType(Enum):
        input2d = 0
        input2halfd = 1
        input3d = 2

    @dataclass
    class ProcessVolumeOpts:
        save_probas_by_class: bool
        debug__save_figs: bool
        override_batch_size: Optional[int]
        save_logs: bool

        @classmethod
        def setup00(cls):
            return cls(
                save_probas_by_class=False,
                debug__save_figs=True,
                save_logs=True,
                override_batch_size=None,
            )

    model_name: str  # the full thing
    model_type: ModelType

    model_shape_min_multiple_requirement: int

    volume_name: str
    volume_version: str

    partition_alias: Optional[str]

    cropping_strategy: CroppingStrategy
    aggregation_strategy: AggregationStrategy

    probabilities_dtype: Type

    opts: ProcessVolumeOpts

    @classmethod
    def setup00_process_test(cls, **kwargs):
        import numpy as np

        kwargs_effective = {
            **dict(
                model_shape_min_multiple_requirement=16,
                partition_alias="test",
                cropping_strategy=ProcessVolumeArgs.CroppingStrategy.maximum_size_reduced_overlap,
                aggregation_strategy=ProcessVolumeArgs.AggregationStrategy.average_probabilities,
                probabilities_dtype=np.float16,
                opts=ProcessVolumeArgs.ProcessVolumeOpts.setup00(),
            ),
            **kwargs,
        }

        return cls(**kwargs_effective)
