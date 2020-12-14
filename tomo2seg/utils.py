from functools import reduce
from operator import mul

import humanize
import numpy as np

from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import InputLayer as KerasInputLayer

from tomo2seg.logger import logger


def get_model_internal_nvoxel_factor(model: KerasModel) -> int:
    """
    If a batch with a single image has N voxels, inside the model,
    because of the model's architecture, the memory must
    hold (N' = f * N) voxels.
    This function returns `f`.
    """

    input_layer = model.layers[0]

    logger.debug(f"{input_layer=}")

    assert isinstance(input_layer, KerasInputLayer), f"{input_layer.__class__=}"

    input_nvoxels = reduce(mul, (x for x in input_layer.input_shape[0][1:]))  # ignore the batch size

    logger.debug(f"{input_nvoxels=}")

    def get_layer_nvoxels(layer) -> int:
        return reduce(mul, (x for x in layer.output_shape[1:]))

    internal_nvoxels = [
        get_layer_nvoxels(l)
        for l in model.layers[1:]
    ]

    max_internal_nvoxels = max(internal_nvoxels)

    logger.debug(f"{max_internal_nvoxels=} ({humanize.intcomma(max_internal_nvoxels)})")

    internal_nvoxel_factor = max_internal_nvoxels / input_nvoxels

    return int(np.ceil(internal_nvoxel_factor))


def fmt_runid(runid: int) -> str:
    s = str(runid)
    return f"{s[:4]}-{s[4:7]}-{s[7:]}"


def parse_runid(runid_str: str) -> int:
    return int("".join(runid_str.split("-")))
