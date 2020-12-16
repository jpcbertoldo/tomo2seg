import time
from copy import copy
from dataclasses import dataclass, field
from enum import Enum
from functools import reduce
from operator import mul
from typing import Type, Optional, Tuple

import numpy as np

from tomo2seg.logger import logger, dict2str


@dataclass
class ProcessVolumeArgs:

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

    runid: int = field(default_factory=lambda: int(time.time()))
    random_state_seed: int = 42  # did you get the reference?

    @classmethod
    def setup00_process_test(cls, **kwargs):
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
        logger.info(f"kwargs_effective=\n{dict2str(kwargs_effective)}")
        return cls(**kwargs_effective)


def get_largest_crop_multiple(volume_shape: Tuple[int, int, int], multiple_of: int) -> Tuple[int, int, int]:
    # noinspection PyTypeChecker
    return tuple(
        int(multiple_of * np.floor(dim / multiple_of))
        for dim in volume_shape
    )


def reduce_dimensions(shape: Tuple[int, int, int], max_nvoxels: int, multiple_of: int) -> Tuple[int, int, int]:

    class FoundDivisibleDim(Exception):
        pass

    shape = list(copy(shape))
    original_shape = tuple(list(copy(shape)))
    nvoxels = int(reduce(mul, shape))

    while nvoxels > max_nvoxels:
        dims_sorted = reversed(np.argsort(shape))

        try:
            for dim in dims_sorted:
                candidate = shape[dim] - multiple_of
                if (candidate % multiple_of) == 0:
                    shape[dim] = int(candidate)
                    raise FoundDivisibleDim()

        except FoundDivisibleDim:
            nvoxels = int(reduce(mul, shape))
            continue

        else:
            raise ValueError(f"Could not find a suitable shape from {original_shape=} {multiple_of=} {max_nvoxels=}. Smallest size found was {shape=} {nvoxels=}")

    # noinspection PyTypeChecker
    return tuple(shape)
