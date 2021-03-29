import operator
from dataclasses import dataclass, field
from enum import Enum
from functools import reduce
from io import BytesIO, IOBase
from pathlib import Path
from typing import Tuple, Union

import yaml
from numpy import ndarray
import numpy as np


class ByteEncoding(Enum):
    littleendian = True
    bigendian = False


def read_raw(
    io_stream: IOBase,
    dtype: Union[np.dtype, str],
    shape: Tuple[int, int, int],
    header_size: int = 0,
    zrange: range = None,
    little_endian: bool = True,
) -> ndarray:
    """

    Adapted from pymicro. Kudos @heprom.

    todo make it possible to convert the annotations into binary individual raws (one for each class)
    todo test me

    Read a volume file stored as a concatenated stack of binary images.
    The raw file is assumed to contain all the bytes sequentially.
    The data type can be set to any numpy type (32 bits float for example).
    Little endian convention is assumed by default.

    ** a note from pymicro **
    If you use this function to read a .edf file written by
    matlab in +y+x+z convention (column major order), you may want to
    use: np.swapaxes(HST_read('file.edf', ...), 0, 1)

    Usage:

        with open("/path/to/file.raw", "wb") as f:
            arr = read_raw(f, "uint8", (32, 32, 32))

    Args:
        :param io_stream: binary file to read io stream.
        :param dtype: numpy data type to use.
        :param shape: a 3-int tuple containing the dimensions along the x, y, and z axes (0 by default).
        :param header_size: number of bytes to skip before reading the payload.
        :param zrange: range of slices to use (by default all the z-slices are read).
        :param little_endian: if you are using big endian convention, set this to False.

    Returns: a 3D np.ndarray.
    """

    if not isinstance(io_stream, IOBase):
        raise TypeError(f"{type(io_stream)=}")

    (nx, ny, nz) = shape

    if zrange is None:
        zrange = range(0, nz)

    zrange = list(zrange)

    nz_effective = len(zrange)

    voxel_size = np.dtype(dtype).itemsize
    nvoxels_zslice = ny * nx

    zslice_size = voxel_size * nvoxels_zslice
    start_position = header_size + zslice_size * zrange[0]

    io_stream.seek(start_position)

    payload = io_stream.read(zslice_size * nz_effective)

    data = np.fromstring(payload, dtype)

    data = np.reshape(data.astype(dtype), (nz_effective, ny, nx), order="C")

    # HP 10/2013 start using proper [x,y,z] data ordering
    data = data.transpose(2, 1, 0)

    if little_endian:
        data.byteswap(True)

    return data


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
    byte_encoding: str = field(init=False, repr=False)

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

        metadata_keys = self._yml_dict.keys()

        for key in [
            "shape",
            "dtype",
            "byte_encoding",
        ]:
            if not key in metadata_keys:
                raise VolumeErrorMetadata(f"{key=} {metadata_keys=}")

        self.dtype = self._yml_dict["dtype"]

        try:
            np.dtype(self.dtype)
        except TypeError:
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

        self.byte_encoding = self._yml_dict["byte_encoding"]

        try:
            ByteEncoding[self.byte_encoding]
        except KeyError:
            raise VolumeErrorMetadata(
                f"{self.byte_encoding=}, must be one of {ByteEncoding._member_names_=}"
            )

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

        with self.raw.open("rb") as f:
            self.array = read_raw(
                io_stream=f,
                zrange=None,
                dtype=self.dtype,
                header_size=0,
                shape=self.shape,
                little_endian=ByteEncoding[self.byte_encoding].value,
            )

        if not self.array.shape == self.shape:
            raise VolumeError()

        if not self.array.dtype == np.dtype(self.dtype):
            raise VolumeError()

    @property
    def dir(self) -> Path:
        return self.file.parent

    @property
    def raw(self) -> Path:
        return self.dir / f"{self.name}.raw"

    def xslice(self, x: int) -> ndarray:
        return self.array[x, :, :]

    def yslice(self, y: int) -> ndarray:
        return self.array[:, y, :]

    def zslice(self, z: int) -> ndarray:
        return self.array[:, :, z]

    def yz(self, x: int) -> ndarray:
        return self.xslice(x)

    def xz(self, y: int) -> ndarray:
        return self.yslice(y)

    def xy(self, z: int) -> ndarray:
        return self.zslice(z)
