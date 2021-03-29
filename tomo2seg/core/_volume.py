import operator
from dataclasses import dataclass, field
from functools import reduce
from pathlib import Path
from typing import Tuple

import yaml
from numpy import ndarray
import numpy as np


# INT_NORMALIZE_FACTORS = {
#     "uint8": 255,
#     "uint16": 65535,
# }


class VolumeError(Exception):
    pass


class VolumeErrorMetadata(VolumeError):
    pass


class VolumeErrorRaw(VolumeError):
    pass


class VolumeErrorMetadataRawMismatch(VolumeError):
    pass


@dataclass
class Volume:
    """A 3D image (pile of 2D images)."""

    file: Path = field(repr=False)

    name: str = field(init=False)
    shape: Tuple[int, int, int] = field(init=False)
    dtype: str = field(init=False)

    array: ndarray = field(init=False, repr=False)

    _yml_dict: dict = field(init=False, repr=False)

    def __post_init__(self):

        if isinstance(self.file, str):
            self.file = Path(self.file)

        self.file = self.file.absolute()

        if not self.file.exists():
            raise VolumeErrorMetadata(f"{str(self.file)=}")

        extension = self.file.suffix

        if not extension in (".yaml", ".yml"):
            raise VolumeErrorMetadata(f"{extension=}")

        self.name = self.file.stem

        with self.file.open("r") as f:
            self._yml_dict = yaml.load(f, Loader=yaml.FullLoader)

        if not "shape" in (yml_dict_keys := self._yml_dict.keys()):
            raise VolumeErrorMetadata(f"{yml_dict_keys=}")

        if not "dtype" in yml_dict_keys:
            raise VolumeErrorMetadata(f"{yml_dict_keys=}")

        self.dtype = self._yml_dict["dtype"]

        try:
            np.dtype(self.dtype)
        except TypeError:
            raise VolumeErrorMetadata(f"{self.dtype=}")

        if not self.dtype in INT_NORMALIZE_FACTORS:
            raise VolumeErrorMetadata(f"{self.dtype=}")

        self.shape = self._yml_dict["shape"]

        if not isinstance(self.shape, list):
            raise VolumeErrorMetadata(f"{type(self.shape)=}")

        self.shape = tuple(self.shape)
        if not (shape_len := len(self.shape)) == 3:
            raise VolumeErrorMetadata(f"{self.shape=} {shape_len=}")

        if not all(isinstance(val, int) for val in self.shape):
            raise VolumeErrorMetadata(f"{self.shape=} {shape_len=}")

        if not all(val > 0 for val in self.shape):
            raise VolumeErrorMetadata(f"{self.shape=} {shape_len=}")

        if not self.raw.exists():
            raise VolumeErrorRaw(f"{self.raw=}")

        raw_size = self.raw.stat().st_size  # size in bytes
        nbytes_per_voxel = np.dtype(self.dtype).itemsize
        nvoxels = reduce(operator.mul, self.shape)
        should_have_bytes = nvoxels * nbytes_per_voxel

        if not raw_size == should_have_bytes:
            raise VolumeErrorMetadataRawMismatch(
                f"(bytes) {raw_size=} {should_have_bytes=} "
            )

    @property
    def dir(self) -> Path:
        return self.file.parent

    @property
    def raw(self) -> Path:
        return self.dir / f"{self.name}.raw"
