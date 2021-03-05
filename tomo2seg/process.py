import time
from copy import copy
from dataclasses import dataclass, field
from enum import Enum
from functools import reduce
from operator import mul
from typing import Type, Optional, Tuple

import numpy as np

from tomo2seg.logger import logger, dict2str


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
