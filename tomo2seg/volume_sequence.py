"""
3D crop generator from a volume.
todo make this module batch-enabled
todo make this all with keras backend
"""

# Standard packages
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field, InitVar
from enum import Enum
from functools import partial
from itertools import product
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Callable, Type, Union, ClassVar
import warnings

# Installed packages
import numpy as np
import scipy as sp
from numpy import ndarray
from numpy.random import RandomState
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates
from tensorflow.keras.utils import Sequence

from tomo2seg.logger import logger
from tomo2seg.data import NORMALIZE_FACTORS


def __get_attr__(name):
    
    if name == "NORMALIZE_FACTORS":
        warnings.warn(f"Please use the version in the module `data.py`.", DeprecationWarning)
        return {
            "uint8": 255,
            "uint16": 65535,
        }
    
    raise AttributeError(f"Module `{__name__}` does not have attribute `{name}`.")


class GT2D(Enum):
    """Canonical 2D Geometric Transformation. Any other one is equivalent to these."""
    identity = 0
    rotation90 = 1
    flip_horizontal = 2
    flip_vertical = 3
    transpose = 4
    flip_horizontal__transpose = 5
    flip_vertical__flip_horizontal = 6
    rotation90__flip_vertical = 7


# a dictionary of transformations so that they can be referenced with a string key
_GT2D_FUNCTIONS: Dict[GT2D, Callable[[ndarray], ndarray]] = {
    GT2D.identity: lambda x: x,  # identity
    GT2D.rotation90: partial(np.rot90, axes=(0, 1), k=1),
    GT2D.flip_horizontal: partial(np.flip, axis=0),
    GT2D.flip_vertical: partial(np.flip, axis=1),
    GT2D.transpose: partial(np.transpose, axes=(1, 0)),
}
    

def _compose(f: Callable, g: Callable) -> Callable:
    """(f, g) -> fog"""
    def composed_function(x):
        return f(g(x))
    return composed_function


# composed transformations - any other combination will result in something equivalent to these here
# todo verify compositions
_GT2D_FUNCTIONS.update({
    GT2D.flip_horizontal__transpose: _compose(
        f=_GT2D_FUNCTIONS[GT2D.flip_horizontal],
        g=_GT2D_FUNCTIONS[GT2D.transpose]
    ),
    GT2D.flip_vertical__flip_horizontal: _compose(
        f=_GT2D_FUNCTIONS[GT2D.flip_vertical],
        g=_GT2D_FUNCTIONS[GT2D.flip_horizontal]
    ),
    GT2D.rotation90__flip_vertical: _compose(
        f=_GT2D_FUNCTIONS[GT2D.rotation90],
        g=_GT2D_FUNCTIONS[GT2D.flip_vertical]
    ),
})

_GT2D_VAL2FUNC = {
    gt.value: func for gt, func in _GT2D_FUNCTIONS.items()
}


_GT2HALFD_FUNCTIONS = deepcopy(_GT2D_FUNCTIONS)
_GT2HALFD_FUNCTIONS.update({
    GT2D.transpose: partial(np.transpose, axes=(1, 0, 2)),
})
_GT2HALFD_FUNCTIONS.update({
    GT2D.flip_horizontal__transpose: _compose(
        f=_GT2HALFD_FUNCTIONS[GT2D.flip_horizontal],
        g=_GT2HALFD_FUNCTIONS[GT2D.transpose]
    ),
})

_GT2HALFD_VAL2FUNC = {
    gt.value: func for gt, func in _GT2HALFD_FUNCTIONS.items()
}


def _get_random_gt_2d(random_state: RandomState) -> GT2D:
    """todo make this batch-enabled"""
    return random_state.choice(GT2D, 1, replace=False)[0]


class GT3D(Enum):
    """
    Canonical 3D Geometric Transformation. Any other one is equivalent to these.
    To find the groups (of equivalent transformation combinations) I exhaustively
    tried all the combinations of 3 with the simple transformations.
    The code to reproduce it is under tomo2seg/scratches/gt3d_groups.py.
    """

    # simple transformations
    identity = 0

    #    x-axis-based
    rotx90 = 10
    rotx180 = 11
    flipx = 12
    transpose_yz = 13

    #    y-axis-based
    roty90 = 20
    roty180 = 21
    flipy = 22
    transpose_xz = 23

    #    z-axis-based
    rotz90 = 30
    rotz180 = 31
    flipz = 32
    transpose_xy = 33

    #    3-axes transpositions
    transpose_yzx = 100
    transpose_zxy = 101

    # combinations of 2 (30)
    rotx90_rotx180 = 1000
    rotx90_flipx = 1001
    rotx90_flipy = 1002
    rotx90_transpose_xz = 1003
    rotx90_rotz90 = 1004
    rotx90_transpose_xy = 1005
    rotx90_transpose_yzx = 1006
    rotx90_transpose_zxy = 1007
    rotx180_flipx = 1008
    rotx180_transpose_xz = 1009
    rotx180_rotz90 = 1010
    rotx180_transpose_xy = 1011
    rotx180_transpose_yzx = 1012
    rotx180_transpose_zxy = 1013
    flipx_transpose_yz = 1014
    flipx_roty90 = 1015
    flipx_flipy = 1016
    flipx_transpose_xz = 1017
    flipx_flipz = 1018
    flipx_transpose_yzx = 1019
    flipx_transpose_zxy = 1020
    roty90_rotx180 = 1021
    roty90_transpose_yz = 1022
    roty90_transpose_xy = 1023
    roty90_transpose_zxy = 1024
    flipy_rotz90 = 1025
    transpose_xz_rotx180 = 1026
    rotz90_roty90 = 1027
    rotz90_flipz = 1028
    transpose_zxy_rotx180 = 1029

    # combinations of 3 (5)
    rotx90_rotx180_flipx = 2000
    rotx90_flipx_flipy = 2001
    rotx90_flipx_transpose_xz = 2002
    rotx90_flipy_rotz90 = 2003
    rotx90_transpose_xz_rotx180 = 2004


# a dictionary of transformations so that they can be referenced with a string key
_GT3D_FUNCTIONS: Dict[GT3D, Callable[[ndarray], ndarray]] = {
    GT3D.identity: lambda x: x,  # identity

    GT3D.rotx90: partial(np.rot90, axes=(1, 2), k=1),
    GT3D.rotx180: partial(np.rot90, axes=(1, 2), k=2),
    GT3D.flipx: partial(np.flip, axis=0),
    GT3D.transpose_yz: partial(np.transpose, axes=(0, 2, 1)),

    GT3D.roty90: partial(np.rot90, axes=(2, 0), k=1),
    GT3D.roty180: partial(np.rot90, axes=(2, 0), k=2),
    GT3D.flipy: partial(np.flip, axis=1),
    GT3D.transpose_xz: partial(np.transpose, axes=(2, 1, 0)),

    GT3D.rotz90: partial(np.rot90, axes=(0, 1), k=1),
    GT3D.rotz180: partial(np.rot90, axes=(0, 1), k=2),
    GT3D.flipz: partial(np.flip, axis=2),
    GT3D.transpose_xy: partial(np.transpose, axes=(1, 0, 2)),

    GT3D.transpose_yzx: partial(np.transpose, axes=(1, 2, 0)),
    GT3D.transpose_zxy: partial(np.transpose, axes=(2, 0, 1)),
}

# composed transformations - any other combination will result in something equivalent to these here
_GT3D_FUNCTIONS.update({
    # combinations of 2 (30)
    GT3D.rotx90_rotx180: _compose(
        f=_GT3D_FUNCTIONS[GT3D.rotx90],
        g=_GT3D_FUNCTIONS[GT3D.rotx180],
    ),
    GT3D.rotx90_flipx: _compose(
        f=_GT3D_FUNCTIONS[GT3D.rotx90],
        g=_GT3D_FUNCTIONS[GT3D.flipx],
    ),
    GT3D.rotx90_flipy: _compose(
        f=_GT3D_FUNCTIONS[GT3D.rotx90],
        g=_GT3D_FUNCTIONS[GT3D.flipy],
    ),
    GT3D.rotx90_transpose_xz: _compose(
        f=_GT3D_FUNCTIONS[GT3D.rotx90],
        g=_GT3D_FUNCTIONS[GT3D.transpose_xz],
    ),
    GT3D.rotx90_rotz90: _compose(
        f=_GT3D_FUNCTIONS[GT3D.rotx90],
        g=_GT3D_FUNCTIONS[GT3D.rotz90],
    ),
    GT3D.rotx90_transpose_xy: _compose(
        f=_GT3D_FUNCTIONS[GT3D.rotx90],
        g=_GT3D_FUNCTIONS[GT3D.transpose_xy],
    ),
    GT3D.rotx90_transpose_yzx: _compose(
        f=_GT3D_FUNCTIONS[GT3D.rotx90],
        g=_GT3D_FUNCTIONS[GT3D.transpose_yzx],
    ),
    GT3D.rotx90_transpose_zxy: _compose(
        f=_GT3D_FUNCTIONS[GT3D.rotx90],
        g=_GT3D_FUNCTIONS[GT3D.transpose_zxy],
    ),
    GT3D.rotx180_flipx: _compose(
        f=_GT3D_FUNCTIONS[GT3D.rotx180],
        g=_GT3D_FUNCTIONS[GT3D.flipx],
    ),
    GT3D.rotx180_transpose_xz: _compose(
        f=_GT3D_FUNCTIONS[GT3D.rotx180],
        g=_GT3D_FUNCTIONS[GT3D.transpose_xz],
    ),
    GT3D.rotx180_rotz90: _compose(
        f=_GT3D_FUNCTIONS[GT3D.rotx180],
        g=_GT3D_FUNCTIONS[GT3D.rotz90],
    ),
    GT3D.rotx180_transpose_xy: _compose(
        f=_GT3D_FUNCTIONS[GT3D.rotx180],
        g=_GT3D_FUNCTIONS[GT3D.transpose_xy],
    ),
    GT3D.rotx180_transpose_yzx: _compose(
        f=_GT3D_FUNCTIONS[GT3D.rotx180],
        g=_GT3D_FUNCTIONS[GT3D.transpose_yzx],
    ),
    GT3D.rotx180_transpose_zxy: _compose(
        f=_GT3D_FUNCTIONS[GT3D.rotx180],
        g=_GT3D_FUNCTIONS[GT3D.transpose_zxy],
    ),
    GT3D.flipx_transpose_yz: _compose(
        f=_GT3D_FUNCTIONS[GT3D.flipx],
        g=_GT3D_FUNCTIONS[GT3D.transpose_yz],
    ),
    GT3D.flipx_roty90: _compose(
        f=_GT3D_FUNCTIONS[GT3D.flipx],
        g=_GT3D_FUNCTIONS[GT3D.roty90],
    ),
    GT3D.flipx_flipy: _compose(
        f=_GT3D_FUNCTIONS[GT3D.flipx],
        g=_GT3D_FUNCTIONS[GT3D.flipy],
    ),
    GT3D.flipx_transpose_xz: _compose(
        f=_GT3D_FUNCTIONS[GT3D.flipx],
        g=_GT3D_FUNCTIONS[GT3D.transpose_xz],
    ),
    GT3D.flipx_flipz: _compose(
        f=_GT3D_FUNCTIONS[GT3D.flipx],
        g=_GT3D_FUNCTIONS[GT3D.flipz],
    ),
    GT3D.flipx_transpose_yzx: _compose(
        f=_GT3D_FUNCTIONS[GT3D.flipx],
        g=_GT3D_FUNCTIONS[GT3D.transpose_yzx],
    ),
    GT3D.flipx_transpose_zxy: _compose(
        f=_GT3D_FUNCTIONS[GT3D.flipx],
        g=_GT3D_FUNCTIONS[GT3D.transpose_zxy],
    ),
    GT3D.roty90_rotx180: _compose(
        f=_GT3D_FUNCTIONS[GT3D.roty90],
        g=_GT3D_FUNCTIONS[GT3D.rotx180],
    ),
    GT3D.roty90_transpose_yz: _compose(
        f=_GT3D_FUNCTIONS[GT3D.roty90],
        g=_GT3D_FUNCTIONS[GT3D.transpose_yz],
    ),
    GT3D.roty90_transpose_xy: _compose(
        f=_GT3D_FUNCTIONS[GT3D.roty90],
        g=_GT3D_FUNCTIONS[GT3D.transpose_xy],
    ),
    GT3D.roty90_transpose_zxy: _compose(
        f=_GT3D_FUNCTIONS[GT3D.roty90],
        g=_GT3D_FUNCTIONS[GT3D.transpose_zxy],
    ),
    GT3D.flipy_rotz90: _compose(
        f=_GT3D_FUNCTIONS[GT3D.flipy],
        g=_GT3D_FUNCTIONS[GT3D.rotz90],
    ),
    GT3D.transpose_xz_rotx180: _compose(
        f=_GT3D_FUNCTIONS[GT3D.transpose_xz],
        g=_GT3D_FUNCTIONS[GT3D.rotx180],
    ),
    GT3D.rotz90_roty90: _compose(
        f=_GT3D_FUNCTIONS[GT3D.rotz90],
        g=_GT3D_FUNCTIONS[GT3D.roty90],
    ),
    GT3D.rotz90_flipz: _compose(
        f=_GT3D_FUNCTIONS[GT3D.rotz90],
        g=_GT3D_FUNCTIONS[GT3D.flipz],
    ),
    GT3D.transpose_zxy_rotx180: _compose(
        f=_GT3D_FUNCTIONS[GT3D.transpose_zxy],
        g=_GT3D_FUNCTIONS[GT3D.rotx180],
    ),

    # combinations of 3 (5)
    GT3D.rotx90_rotx180_flipx: _compose(
        f=_GT3D_FUNCTIONS[GT3D.rotx90],
        g=_compose(
            f=_GT3D_FUNCTIONS[GT3D.rotx180],
            g=_GT3D_FUNCTIONS[GT3D.flipx],
        ),
    ),
    GT3D.rotx90_flipx_flipy:  _compose(
        f=_GT3D_FUNCTIONS[GT3D.rotx90],
        g=_compose(
            f=_GT3D_FUNCTIONS[GT3D.flipx],
            g=_GT3D_FUNCTIONS[GT3D.flipy],
        ),
    ),
    GT3D.rotx90_flipx_transpose_xz:  _compose(
        f=_GT3D_FUNCTIONS[GT3D.rotx90],
        g=_compose(
            f=_GT3D_FUNCTIONS[GT3D.flipx],
            g=_GT3D_FUNCTIONS[GT3D.transpose_xz],
        ),
    ),
    GT3D.rotx90_flipy_rotz90:  _compose(
        f=_GT3D_FUNCTIONS[GT3D.rotx90],
        g=_compose(
            f=_GT3D_FUNCTIONS[GT3D.flipy],
            g=_GT3D_FUNCTIONS[GT3D.rotz90],
        ),
    ),
    GT3D.rotx90_transpose_xz_rotx180:  _compose(
        f=_GT3D_FUNCTIONS[GT3D.rotx90],
        g=_compose(
            f=_GT3D_FUNCTIONS[GT3D.transpose_xz],
            g=_GT3D_FUNCTIONS[GT3D.rotx180],
        ),
    ),
})

_GT3D_VAL2FUNC = {
    gt.value: func for gt, func in _GT3D_FUNCTIONS.items()
}


def _get_random_gt_3d(random_state: RandomState) -> GT3D:
    """todo make this batch-enabled"""
    return random_state.choice(GT3D, 1, replace=False)[0]


@dataclass
class ET:
    """ElasticTransformation. Stocks the 3D displacement of each of the 3D crop's 8 corners."""
    # c = corner
    c000: Tuple[float, float, float] = (0., 0., 0,)
    c100: Tuple[float, float, float] = (0., 0., 0,)
    c010: Tuple[float, float, float] = (0., 0., 0,)
    c110: Tuple[float, float, float] = (0., 0., 0,)
    c001: Tuple[float, float, float] = (0., 0., 0,)
    c101: Tuple[float, float, float] = (0., 0., 0,)
    c011: Tuple[float, float, float] = (0., 0., 0,)
    c111: Tuple[float, float, float] = (0., 0., 0,)


@dataclass
class GridPositionGenerator(ABC):
    
    x_range: Tuple[int, int]
    y_range: Tuple[int, int]
    z_range: Tuple[int, int]

    def __post_init__(self):
        for axis, axis_range in zip(range(3), self.axes_ranges):
            assert axis_range[0] <= axis_range[1], f"{axis=} {axis_range=}"
        
        import functools
        import operator
        import humanize
        npositions = functools.reduce(operator.mul, self.axes_range_sizes)
        logger.debug(f"{self.__class__.__name__} ==> {npositions=} ({humanize.intcomma(npositions)})")

    @property
    def axes_ranges(self) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        return self.x_range, self.y_range, self.z_range

    @property
    def axes_range_sizes(self) -> Tuple[int, int, int]:
        return tuple(stop - start for start, stop in self.axes_ranges)

    def get(self, n: int) -> ndarray:  # (n, 3)
        assert n > 0
        grid_positions = self._concrete_getitem(n)
        assert grid_positions.shape == (n, 3), f"{self.__class__.__name__=} {n=} {grid_positions.shape=}"
        return grid_positions

    @abstractmethod
    def _concrete_getitem(self, n: int) -> ndarray:  # (n, 3)
        pass

    @classmethod
    def build_from_volume_crop_shapes(
            cls, volume_shape: Tuple[int, int, int], crop_shape: Tuple[int, int, int], *args, **kwargs
    ):
        if "x_range" in kwargs or "y_range" in kwargs or "z_range" in kwargs:
            raise ValueError(
                f"Trying to build {cls.__name__} from volume and crop shapes but range values were given in kwargs. "
                f"Please remove them. {list(kwargs.keys())=}"
            )

        for axis, volume_axis_size, crop_axis_size in zip(range(3), volume_shape, crop_shape):
            assert 0 < crop_axis_size <= volume_axis_size, f"{axis=} {crop_axis_size=} {volume_axis_size=}"

        ranges = {
            "x_range": (0, volume_shape[0] - crop_shape[0] + 1),
            "y_range": (0, volume_shape[1] - crop_shape[1] + 1),
            "z_range": (0, volume_shape[2] - crop_shape[2] + 1),
        }
        logger.info(f"Built {cls.__name__} from {volume_shape=} and {crop_shape=} ==> {ranges}")

        # noinspection PyArgumentList
        return cls(
            *args,
            **{**kwargs, **ranges}
        )


@dataclass
class UniformGridPosition(GridPositionGenerator):

    random_state: RandomState

    def _concrete_getitem(self, n: int) -> ndarray:  # (n, 3)
        return np.stack([
            self.random_state.randint(
                low=range_[0],
                high=range_[1],
                size=n
            )
            for range_ in self.axes_ranges  # x, y, z-range
        ], axis=-1)


@dataclass
class SequentialGridPosition(GridPositionGenerator):

    n_steps_x: int
    n_steps_y: int
    n_steps_z: int

    @property
    def axes_n_steps(self) -> Tuple[int, int, int]:
        return self.n_steps_x, self.n_steps_y, self.n_steps_z

    def __post_init__(self):
        for axis, axis_n_steps, axis_range_size in zip(range(3), self.axes_n_steps, self.axes_range_sizes):
            assert 0 < axis_n_steps <= axis_range_size, f"{axis=} {axis_n_steps=} {axis_range_size=}"

        self.positions = np.array([
            [int(x), int(y), int(z)]
            for z, y, x in product(
                np.linspace(self.z_range[0], self.z_range[1] - 1, self.n_steps_z, endpoint=True),
                np.linspace(self.y_range[0], self.y_range[1] - 1, self.n_steps_y, endpoint=True),
                np.linspace(self.x_range[0], self.x_range[1] - 1, self.n_steps_x, endpoint=True),
            )
        ])
        logger.info(f"The {self.__class__.__name__} has {len(self.positions)=} different positions (therefore crops).")
        self.current_position = 0

    def __len__(self):
        return len(self.positions)

    def _concrete_getitem(self, n: int) -> ndarray:
        positions = self.positions[self.current_position:(new_current := self.current_position + n)]
        self.current_position = new_current if new_current < len(self.positions) else 0
        if len(positions) < n:
            positions = np.r_[positions, (n - len(positions)) * [positions[-1, :]]]
        return positions

    @classmethod
    def build_min_overlap(cls, volume_shape: Tuple[int, int, int], crop_shape: Tuple[int, int, int], **n_steps_kwargs):
        n_steps = dict(
            n_steps_x=int(np.ceil(volume_shape[0] / crop_shape[0])),
            n_steps_y=int(np.ceil(volume_shape[1] / crop_shape[1])),
            n_steps_z=int(np.ceil(volume_shape[2] / crop_shape[2])),
        )
        logger.info(f"Building {cls.__name__} with minimal overlap (smallest n_steps in each directions) {n_steps=}.")
        if len(n_steps_kwargs) > 0:
            n_steps = {**n_steps, **n_steps_kwargs}
            logger.warning(f"{n_steps_kwargs=} was given --> effective {n_steps=}")

        # noinspection PyArgumentList
        return cls.build_from_volume_crop_shapes(
            volume_shape=volume_shape,
            crop_shape=crop_shape,
            **n_steps,
        )

    @classmethod
    def build_all_positions(cls, volume_shape: Tuple[int, int, int], crop_shape: Tuple[int, int, int]):
        n_steps = dict(
            n_steps_x=volume_shape[0] - crop_shape[0] + 1,
            n_steps_y=volume_shape[1] - crop_shape[1] + 1,
            n_steps_z=volume_shape[2] - crop_shape[2] + 1,
        )
        # noinspection PyArgumentList
        return cls.build_from_volume_crop_shapes(
            volume_shape=volume_shape,
            crop_shape=crop_shape,
            **n_steps,
        )


@dataclass
class VoxelwiseProbabilityGridPotion(GridPositionGenerator):
    """
    Each position in the grid has its own probability.
    Get a tuple (i, j) randomly with the marginal probability on axis (0, 1),
    then use the conditional probability P[k|i,j] to pick the k value.
    Picking (i, j, k) at once is too slow.

    Attention: the arg `probabilities_volume` might be modified, so only pass copies!
    """

    probabilities_volume: ndarray
    random_state: RandomState

    def __post_init__(self):
        self.vol_shape = vol_shape = self.probabilities_volume.shape
        for axis, (start, stop), vol_axis_size in zip(range(3), self.axes_ranges, vol_shape):
            assert (range_size := stop - start) == vol_axis_size, f"Incompatible on {axis=} {range_size=} {vol_axis_size=}"

        # marginal probabilities of (i, j)
        self.ij_probas = self.probabilities_volume.sum(axis=2, keepdims=True)
        # doing it twice makes it more stable
        self.ij_probas /= self.ij_probas.sum()

        # probabilities P[i,j,k] become P[k|i,j]
        self.probabilities_volume /= self.ij_probas
        # doing it twice makes it more stable
        self.probabilities_volume /= self.probabilities_volume.sum(axis=2, keepdims=True)

        # get rid of unnecessary dimensions
        self.ij_probas = self.ij_probas[:, :, 0]

        # this is the tolerance of the function choice
        # https://github.com/numpy/numpy/blob/c2b6ab9924271b96d3c783f7818723a1bb8f511a/numpy/random/mtrand/mtrand.pyx#L1093
        assert np.isclose((ij_probas_sum := self.ij_probas.sum()), 1., atol=1e-8), f"{ij_probas_sum=}"
        assert np.all(np.isclose(self.probabilities_volume.sum(axis=2), 1., atol=1e-8)), "oops"

        self.ij_probas = self.ij_probas.ravel()
        self.ij_sequentials = np.arange(vol_shape[0] * vol_shape[1])
        self.k_sequentials = np.arange(vol_shape[2])

    def _concrete_getitem(self, n_positions: int) -> ndarray:
        ns = self.random_state.choice(
            self.ij_sequentials,
            p=self.ij_probas,
            replace=True,
            size=n_positions
        )
        ijs = np.c_[ns // self.vol_shape[1], ns % self.vol_shape[1]]
        return np.c_[ijs[:, 0], ijs[:, 1], [
            self.random_state.choice(
                self.k_sequentials,
                p=self.probabilities_volume[i, j, :],
                replace=True,
                size=1
            )
            for i, j in ijs
        ]]


@dataclass
class VoxelwiseProbabilityGridPotion2(GridPositionGenerator):
    """
    Each position in the grid has its own probability.
    Get a position `i` randomly with the marginal probability on axis (0,),
    then use the conditional probability P[j|i] to pick the `j` value,
    then P[k|i,j] to pick `k`.
    Picking (i, j, k) at once is too slow.

    Attention: the arg `probabilities_volume` might be modified, so only pass copies!
    """

    probabilities_volume: ndarray
    random_state: RandomState

    def __post_init__(self):
        self.vol_shape = vol_shape = self.probabilities_volume.shape
        for axis, (start, stop), vol_axis_size in zip(range(3), self.axes_ranges, vol_shape):
            assert (range_size := stop - start) == vol_axis_size, f"Incompatible on {axis=} {range_size=} {vol_axis_size=}"

        # marginal probabilities of (i, j)
        self.ij_probas = self.probabilities_volume.sum(axis=2, keepdims=True)
        # doing it twice makes it more stable
        self.ij_probas /= self.ij_probas.sum()

        # probabilities P[i,j,k] become P[k|i,j]
        self.probabilities_volume /= self.ij_probas
        # doing it twice makes it more stable
        self.probabilities_volume /= self.probabilities_volume.sum(axis=2, keepdims=True)

        # get rid of unnecessary dimensions
        self.ij_probas = self.ij_probas[:, :, 0]

        # marginal probabilities of `i`
        self.i_probas = self.ij_probas.sum(axis=1, keepdims=True)
        # doing it twice makes it more stable
        self.i_probas /= self.i_probas.sum()

        # the marginal probabilities P[i, j] become P[j|i]
        self.ij_probas /= self.i_probas
        # doing it twice makes it more stable
        self.ij_probas /= self.ij_probas.sum(axis=1, keepdims=True)

        # get rid of unnecessary dimensions
        self.i_probas = self.i_probas[:, 0]

        # this is the tolerance of the function choice
        # https://github.com/numpy/numpy/blob/c2b6ab9924271b96d3c783f7818723a1bb8f511a/numpy/random/mtrand/mtrand.pyx#L1093
        assert np.isclose((i_probas_sum := self.i_probas.sum()), 1., atol=1e-8), f"{i_probas_sum=}"
        assert np.all(np.isclose(self.ij_probas.sum(axis=1), 1., atol=1e-8)), "oops"
        assert np.all(np.isclose(self.probabilities_volume.sum(axis=2), 1., atol=1e-8)), "oops"

        self.i_sequentials = np.arange(vol_shape[0])
        self.j_sequentials = np.arange(vol_shape[1])
        self.k_sequentials = np.arange(vol_shape[2])

    def _concrete_getitem(self, n_positions: int) -> ndarray:
        is_ = self.random_state.choice(
            self.i_sequentials,
            p=self.i_probas,
            replace=True,
            size=n_positions
        )
        ijs = np.c_[is_, [
            self.random_state.choice(
                self.j_sequentials,
                p=self.ij_probas[i, :],
                replace=True,
                size=1,
            )
            for i in is_
        ]]

        return np.c_[ijs[:, 0], ijs[:, 1], [
            self.random_state.choice(
                self.k_sequentials,
                p=self.probabilities_volume[i, j, :],
                replace=True,
                size=1
            )
            for i, j in ijs
        ]]


@dataclass
class ProbabilityField3D(ABC):
    """
    Each position of the volume has a probability distribution over the possible values.
    It supposes a regular grid of unitary steps in every axis.
    """
    random_state: Optional[RandomState]  # some concrete classes are deterministic

    # they are None because they might be inferred from the grid position generator
    # but if the latter is not given then the user must provide these
    x_range: Optional[Tuple[int, int]] = None
    y_range: Optional[Tuple[int, int]] = None
    z_range: Optional[Tuple[int, int]] = None

    grid_position_generator: InitVar[GridPositionGenerator] = None

    def __post_init__(self, grid_position_generator: GridPositionGenerator):
        if grid_position_generator is not None:
            logger.warning(
                f"Initializing {self.__class__.__name__} with a {grid_position_generator.__class__.__name__}.\n"
                f"The {{x, y, z}}_range values will be overwritten."
            )
            self.x_range = grid_position_generator.x_range
            self.y_range = grid_position_generator.y_range
            self.z_range = grid_position_generator.z_range

        else:
            assert self.x_range is not None, f"{self.x_range=}"
            assert self.y_range is not None, f"{self.y_range=}"
            assert self.z_range is not None, f"{self.z_range=}"

    @property
    def axes_ranges(self) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        return self.x_range, self.y_range, self.z_range

    def __getitem__(self, coordinates3d: Tuple[int, int, int]):
        for axis, coord, axis_lim in zip(range(3), coordinates3d, [self.x_range, self.y_range, self.z_range]):
            assert axis_lim[0] <= coord < axis_lim[1], f"{coord=} on {axis=} out of range {axis_lim=}"
        return self._concrete_getitem(*coordinates3d)

    @abstractmethod
    def _concrete_getitem(self, x: int, y: int, z: int):
        """Given a position in a volume, return some object."""
        pass


@dataclass
class GTConstantEverywhere(ProbabilityField3D):

    gt: Union[GT2D, GT3D] = GT3D.identity
    random_state: Optional[RandomState] = None

    def _concrete_getitem(self, *_):
        return self.gt

    @classmethod
    def build_gt2d_identity(cls, **kwargs):
        # noinspection PyArgumentList
        return cls(gt=GT2D.identity, **kwargs)

    @classmethod
    def build_gt3d_identity(cls, **kwargs):
        # noinspection PyArgumentList
        return cls(gt=GT3D.identity, **kwargs)


@dataclass
class GTUniformEverywhere(ProbabilityField3D):

    gt_type: Type = GT3D  # GT2D or GT3D

    def _concrete_getitem(self, x: int, y: int, z: int):
        """
        Every position has a uniform probability over all the possible transformations.
        todo make this batch-enabled
        """

        if self.gt_type == GT2D:
            return _get_random_gt_2d(random_state=self.random_state)

        elif self.gt_type == GT3D:
            return _get_random_gt_3d(random_state=self.random_state)

        else:
            raise ValueError(f"Unknown type of Geometric Transformation. {self.gt_type=}")

    @classmethod
    def build_2d(cls, *args, **kwargs):
        # noinspection PyArgumentList
        return cls(
            *args,
            gt_type=GT2D,
            **kwargs,
        )

    @classmethod
    def build_3d(cls, *args, **kwargs):
        # noinspection PyArgumentList
        return cls(
            *args,
            gt_type=GT3D,
            **kwargs,
        )


@dataclass
class GTUniformEverywhere2(ProbabilityField3D):
    
    gt_type: Type = GT3D  # GT2D or GT3D
    gt_name_list: List[str] = field(default_factory=lambda: []) # the names
        
    def __post_init__(self, *args, **kwargs):
        # noinspection PyArgumentList
        super().__post_init__(*args, **kwargs)
        
        for gt_name in self.gt_name_list:
            self.gt_type[gt_name]
        
        logger.debug(f"{len(self.gt_name_list)=}")
 
    def _concrete_getitem(self, x: int, y: int, z: int):
        """
        todo make this batch-enabled
        """
        
        gt_name = self.random_state.choice(self.gt_name_list, 1, replace=False)[0]
        return self.gt_type[gt_name]
    
    def build_all(cls, gt_type: Type, *args, **kwargs):
        return cls(
            *args,
            gt_type=gt_type,
            gt_name_list=[
                s.lstrip(f"{gt_type.__name__}.") 
                for s in map(str, gt_type)
            ],
            **kwargs,
        )
    
    def gt_no_transpose_rot(cls, gt_type: Type, *args, **kwargs):
        return cls(
            *args,
            gt_type=gt_type,
            gt_name_list=[
                s.lstrip(f"{gt_type.__name__}.") 
                for s in map(str, gt_type)
                # todo improve this to make sure the end shape is the same
                if "transpose" not in s
                and not ("rot" in s and "90" in s)
            ],
            **kwargs,
        )


    
@dataclass
class VSConstantEverywhere(ProbabilityField3D):
    """Values shift is always the same everywhere."""

    shift: float = 0.  # shift on the normalized range [0, 1]
    random_state: Optional[RandomState] = None

    def __post_init__(self, *args, **kwargs):
        # noinspection PyArgumentList
        super().__post_init__(*args, **kwargs)
        assert -1 < self.shift < 1, f"{self.shift=}"

    def _concrete_getitem(self, *_):
        return self.shift

    @classmethod
    def build_no_shift(cls, **kwargs):
        # noinspection PyArgumentList
        return cls(shift=0., **kwargs)


@dataclass
class VSUniformEverywhere(ProbabilityField3D):
    """Values shift is a random value in a uniform interval [shift_min, shift_max] everywhere."""

    shift_min: float = 0.  # shifts on the normalized range [0, 1]
    shift_max: float = 0.

    def __post_init__(self, *args, **kwargs):
        # noinspection PyArgumentList
        super(VSUniformEverywhere, self).__post_init__(*args, **kwargs)
        assert -1 < self.shift_min < 1, f"{self.shift_min=}"
        assert -1 < self.shift_max < 1, f"{self.shift_max=}"
        assert self.shift_min < self.shift_max, f"{self.shift_min=} {self.shift_max=}"

    def _concrete_getitem(self, *_):
        return self.random_state.uniform(
            low=self.shift_min,
            high=self.shift_max,
            size=None,  # scalar is returned
        )

    @classmethod
    def build_plus_or_mines(cls, shift: float, **kwargs):
        return cls(shift_min=-shift, shift_max=shift, **kwargs)


def uniform_cuboid(
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    z_range: Tuple[float, float],
    random_state: RandomState,
) -> Tuple[float, float, float]:
    """
    Return a 3D coordinate inside the cuboid:
    [cuboid_x_range[0], cuboid_x_range[1]]
    X [cuboid_y_range[0], cuboid_y_range[1]]
    X [cuboid_z_range[0], cuboid_z_range[1]].
    """
    # noinspection PyTypeChecker
    return tuple(
        (
            random_state.uniform(
                low=range_[0], high=range_[1], size=None,
            )
            if range_[1] > range_[0]
            else 0
        )
        for range_ in (x_range, y_range, z_range)
    )


@dataclass
class ET3DConstantEverywhere(ProbabilityField3D):

    displacement: Optional[ET] = None
    random_state: Optional[RandomState] = None

    def _concrete_getitem(self, *_):
        return self.displacement

    @classmethod
    def build_no_displacement(cls, **kwargs):
        # noinspection PyArgumentList
        return cls(displacement=None, **kwargs)


@dataclass
class ET3DUniformCuboidAlmostEverywhere(ProbabilityField3D):
    """
    For each position in the 3D field, return 8 values, which correspond to the displacements of
    the corners of the 3D crop for an ET. Remember that the 3D crops are referenced by their x0y0z0
    corner, which is the (x, y, z) sent to this field.
    For a given position, each of the 8 corners have a uniform cuboid around it
    of size [-cuboid_size[i], +cuboid_size[i]] ** 3, except when the displacement will cause the ET crop
    to fall out of the volume's limits.

    cuboid_size[i] = crop_shape[i] * elastic_param
    """

    crop_shape: Tuple[int, int, int] = None
    cuboid_shape: Tuple[float, float, float] = None

    spline_order: int = 3  # default copied from etienne's code

    # the limits of the data grid, i.e. the elastic crop cannot go outside of this range
    # if crop_source_volume_shape is given, these will be overwritten assuming
    # limits [0, shape[idx]] (i.e. the whole volume)
    crop_xlim: Optional[Tuple[int, int]] = None
    crop_ylim: Optional[Tuple[int, int]] = None
    crop_zlim: Optional[Tuple[int, int]] = None
    crop_source_volume_shape: InitVar[Optional[Tuple[int, int, int]]] = None

    def __post_init__(self, *args, **kwargs):
        args, crop_source_volume_shape = args[:-1], args[-1]
        # noinspection PyArgumentList
        super(ET3DUniformCuboidAlmostEverywhere, self).__post_init__(*args, **kwargs)

        if crop_source_volume_shape is not None:
            logger.warning(
                f"Initializing {self.__class__.__name__} with a {crop_source_volume_shape=}.\n"
                f"The crop_{{x, y, z}}lim values will be overwritten with (0, crop_source_volume_shape[{{0, 1, 2}}])."
            )
            self.crop_xlim = (0, crop_source_volume_shape[0])
            self.crop_ylim = (0, crop_source_volume_shape[1])
            self.crop_zlim = (0, crop_source_volume_shape[2])
        else:
            assert self.crop_xlim is not None, f"{self.crop_xlim=}"
            assert self.crop_ylim is not None, f"{self.crop_ylim=}"
            assert self.crop_zlim is not None, f"{self.crop_zlim=}"

        # i have to do this because i cannot have non-default attributes
        assert self.cuboid_shape is not None
        assert self.crop_shape is not None

    def _concrete_getitem(self, x0: int, y0: int, z0: int) -> ET:
        """(x0, y0, z0) is the 000 corner of a 3D crop. todo make this batch-enabled!"""
        uniform_cuboid_ = partial(uniform_cuboid, random_state=self.random_state)

        displacements: Dict[str, Tuple[float, float, float]] = {}

        # the crop's corners
        for corner_x_, corner_y_, corner_z_ in product((0, 1), (0, 1), (0, 1)):

            # coordinates of the crop's corner
            x, y, z = (
                x0 + corner_x_ * self.crop_shape[0],
                y0 + corner_y_ * self.crop_shape[1],
                z0 + corner_z_ * self.crop_shape[2],
            )

            # the max/min here make sure that the elastic crop doesn't go out of the data grid
            transformed_corner3d = uniform_cuboid_(
                x_range=(
                    max(self.crop_xlim[0], x - self.cuboid_shape[0]),
                    min(self.crop_xlim[1], x + self.cuboid_shape[0])
                ),
                y_range=(
                    max(self.crop_ylim[0], y - self.cuboid_shape[1]),
                    min(self.crop_ylim[1], y + self.cuboid_shape[1])
                ),
                z_range=(
                    max(self.crop_zlim[0], z - self.cuboid_shape[2]),
                    min(self.crop_zlim[1], z + self.cuboid_shape[2])
                ),
            )

            displacements[f"c{corner_x_}{corner_y_}{corner_z_}"] = (
                transformed_corner3d[0] - x,
                transformed_corner3d[1] - y,
                transformed_corner3d[2] - z,
            )

        # noinspection PyArgumentList
        return ET(**displacements)

    @classmethod
    def build_half_voxel_cuboid(cls, *args, **kwargs):
        """
        Each voxel's probability cuboid is of the size of a voxel (half to each direction)
        in all the axes, so there is no overlap between each voxel's domain.
        """
        # noinspection PyArgumentList
        return cls(*args, cuboid_shape=(.5, .5, .5), **kwargs)


@dataclass
class MetaCrop3D:
    csv_header: ClassVar[str] = "x,y,z,et,gt_type,gt,vs,is_2halfd"

    x: slice
    y: slice
    z: slice
    et: Optional[ET]
    gt: Union[GT2D, GT3D]
    vs: float  # shift on the normalized range [0, 1]
    
    is_2halfd: bool = False

    @property
    def slice(self) -> Tuple[slice, slice, slice]:
        return self.x, self.y, self.z

    @property
    def shape(self) -> Tuple[int, int, int]:
        # noinspection PyTypeChecker
        return tuple(
            axis.stop - axis.start
            for axis in self.slice
        )

    def get_corner_coords(self, x_: int, y_: int, z_: int) -> Tuple[int, int, int]:
        assert x_ in (0, 1)
        assert y_ in (0, 1)
        assert z_ in (0, 1)
        return (
            self.x.stop if x_ == 1 else self.x.start,
            self.y.stop if y_ == 1 else self.y.start,
            self.z.stop if z_ == 1 else self.z.start,
        )

    def get_et_corner_coords(self,  x_: int, y_: int, z_: int) -> Tuple[float, float, float]:
        assert x_ in (0, 1), f"{x_=}"
        assert y_ in (0, 1), f"{y_=}"
        assert z_ in (0, 1), f"{z_=}"
        # noinspection PyTypeChecker
        return tuple(
            coord + displacement
            for coord, displacement in zip(
                self.get_corner_coords(x_, y_, z_),
                getattr(self.et, f"c{x_}{y_}{z_}")
            )
        )

    def is2d_on(self, axis_idx: int) -> bool:
        other_axis_0, other_axis_1 = [i for i in range(3) if i != axis_idx]
        return (
            (self.shape[axis_idx] == 1)
            and (self.shape[other_axis_0] > 1)
            and (self.shape[other_axis_1] > 1)
        )

    @property
    def flat_axis(self) -> int:
        assert self.is2d, f"{self.is2d=} {self.shape=}"
        return [idx for idx in range(3) if self.is2d_on(idx)][0]

    @property
    def is2d(self):
        return any(self.is2d_on(axis_idx) for axis_idx in range(3))

    def to_csv_line(self) -> str:
        return f"{self.x},{self.y},{self.z},{repr(self.et)},{self.gt.__class__.__name__},{self.gt.name},{self.vs},{self.is_2halfd}"


def _labels_single2multi_channel(im: ndarray, labels: List[int]) -> np.ndarray:
    """ Extract labels of interest from input segmentation """
    assert im.ndim == 3
    im_out = np.zeros((im.shape[0], im.shape[1], im.shape[2], len(labels)))
    for final_label, label in enumerate(labels):
        # noinspection PyUnresolvedReferences
        im_out[:, :, :, final_label] = (im == label).astype(np.uint8)
    return im_out


def meta2crop(
        meta_crop: MetaCrop3D,
        volume: ndarray,
        is_label: bool,
        interpolation: str,
        spline_order: int = None,
) -> np.ndarray:
    """
    todo? make this function multichannel enabled
    :param is_label:
    :param spline_order:
    :param volume:
    :param meta_crop:
    :param interpolation: "spline", "nearest"
    :return:
    """
    assert (is_label and interpolation == "nearest") or not is_label, \
        "Labels should only be interpolated with he nearest neighbor because it is a categorical value."

    assert not (meta_crop.et is not None and interpolation == "spline" and spline_order is None), \
        "If elastic transformation is used then the interpolation spline order has to be given."

    if meta_crop.et is None:
        crop = volume[meta_crop.slice].copy()
    else:
        xbin = (0,) if meta_crop.is2d_on(0) else (0, 1)
        ybin = (0,) if meta_crop.is2d_on(1) else (0, 1)
        zbin = (0,) if meta_crop.is2d_on(2) else (0, 1)
        corners_binaries = list(product(xbin, ybin, zbin))

        ijk_grid_corners = np.array([
            [
                0 if c == 0 else s - 1
                for c, s in zip(corner, meta_crop.shape)
            ]
            for corner in corners_binaries
        ])  # 4x3 or 8x3

        ijk_grid_points = np.array(list(product(
            np.arange(meta_crop.shape[0]),
            np.arange(meta_crop.shape[1]),
            np.arange(meta_crop.shape[2]),
        )))

        et_grid_corner_coords = np.array([
            meta_crop.get_et_corner_coords(ci_, cj_, ck_)
            for ci_, cj_, ck_ in corners_binaries
        ])  # 4x3 or 8x3

        if meta_crop.is2d:
            axes = [0, 1, 2]
            axes.pop(meta_crop.flat_axis)
            ijk_grid_corners = ijk_grid_corners[:, axes]
            ijk_grid_points = ijk_grid_points[:, axes]
            et_grid_corner_coords = et_grid_corner_coords[:, axes]
            # et_grid_corner_coords[:, meta_crop.flat_axis] = meta_crop.get_corner_coords(0, 0, 0)[meta_crop.flat_axis]

        def get_et_grid_coords(axis):
            return sp.interpolate.griddata(
                points=ijk_grid_corners,
                values=et_grid_corner_coords[:, axis].ravel().copy(),
                method="linear",
                xi=ijk_grid_points,
            )  # n_grid_points x 1

        et_grid_coords = np.stack(
            [
                meta_crop.get_corner_coords(0, 0, 0)[meta_crop.flat_axis] * np.ones(ijk_grid_points.shape[0])
                if meta_crop.is2d_on(dim) else
                get_et_grid_coords(dim)
                for dim in range(3)
            ], axis=-1
        )  # n_grid_points x 3

        if interpolation == "spline":
            # todo test if t will be too slow cuz it might do a spline for the entire volume (?)
            crop = map_coordinates(
                input=volume,
                coordinates=et_grid_coords.T,
                # todo check how this works, can it break it order=3 and the crop is in the corner?
                # todo visualize the effect with crops at the borders because this will search for 3rd order derivatives
                order=spline_order,
                mode="mirror",
            )
        elif interpolation == "nearest":
            # todo keep this somewhere to avoid generating?
            crop = sp.interpolate.interpn(
                points=list(
                    np.arange(volume.shape[dim])
                    for dim in range(3)
                ),
                values=volume,
                xi=et_grid_coords,
                method="nearest",
            )
        else:
            raise NotImplementedError(f"{interpolation=} is not supported.")

        crop = crop.reshape(*meta_crop.shape)

    # geometric_transformation
    if isinstance(meta_crop.gt, GT2D):
        if meta_crop.is_2halfd:
            func = _GT2HALFD_VAL2FUNC[meta_crop.gt.value]
        else:
            func = _GT2D_VAL2FUNC[meta_crop.gt.value]
        
    else:
        func = _GT3D_VAL2FUNC[meta_crop.gt.value]

    if meta_crop.is2d_on(0):
        sx = crop.shape[1]
        sy = crop.shape[2]
        crop = func(crop.reshape(sx, sy)).reshape(1, sx, sy)
    elif meta_crop.is2d_on(1):
        sx = crop.shape[0]
        sy = crop.shape[2]
        crop = func(crop.reshape(sx, sy)).reshape(sx, 1, sy)
    elif meta_crop.is2d_on(2):
        sx = crop.shape[0]
        sy = crop.shape[1]
        crop = func(crop.reshape(sx, sy)).reshape(sx, sy, 1)
    else:
        crop = func(crop)

    if is_label:
        return crop.astype(int)
    else:
        crop += meta_crop.vs
        # the transformations might result in something out of the normalized range
        crop = np.clip(crop, a_min=0., a_max=1.)
        return crop


@dataclass
class MetaCrop3DGenerator:
    volume_shape: Tuple[int, int, int]
    crop_shape: Tuple[int, int, int]

    x0y0z0_generator: GridPositionGenerator
    et_field: ProbabilityField3D
    gt_field: ProbabilityField3D
    vs_field: ProbabilityField3D
    
    is_2halfd: bool = False

    def __post_init__(self):

        assert all(0 < s <= axis_size for s, axis_size in zip(self.crop_shape, self.volume_shape))

        # make sure it wont try to slice something beyond the 3d array
        # the crops are referenced by their 000 corner (i.e. the lowest x, y, z)
        for axis_idx in range(3):
            axis_range = self.x0y0z0_generator.axes_ranges[axis_idx]
            volume_axis_size = self.volume_shape[axis_idx]
            crop_axis_size = self.crop_shape[axis_idx]
            assert (
                    0 <= axis_range[0] and axis_range[1] <= volume_axis_size - crop_axis_size + 1
            ), f"{axis_range=} of {axis_idx=} is incompatible with {volume_axis_size=} and { crop_axis_size=}"

        for axis_idx in range(3):
            xyz_range = self.x0y0z0_generator.axes_ranges[axis_idx]
            et_range = self.et_field.axes_ranges[axis_idx]
            gt_range = self.gt_field.axes_ranges[axis_idx]
            vs_range = self.vs_field.axes_ranges[axis_idx]
            # the grid position generator's range (xyz_range) is the reference because it was verified above
            assert (xyz_range == et_range), f"Incompatible {et_range=} with {xyz_range=}"
            assert (xyz_range == gt_range), f"Incompatible {gt_range=} with {xyz_range=}"
            assert (xyz_range == vs_range), f"Incompatible {vs_range=} with {xyz_range=}"

    def get(self, n: int) -> List[MetaCrop3D]:
        x0y0z0_array = self.x0y0z0_generator.get(n)  # n x 3
        return [
            MetaCrop3D(
                x=slice(x, x + self.crop_shape[0]),
                y=slice(y, y + self.crop_shape[1]),
                z=slice(z, z + self.crop_shape[2]),
                et=self.et_field[x, y, z],
                gt=self.gt_field[x, y, z],
                vs=self.vs_field[x, y, z],
                is_2halfd=self.is_2halfd,
            )
            for x, y, z in x0y0z0_array
        ]
    
    @classmethod
    def build_setup_train00(
        cls, 
        volume_shape: Tuple[int, int, int], 
        crop_shape: Tuple[int, int, int], 
        common_random_state_seed: int,
        gt_type: Type,
        is_2halfd: bool,
        data_original_dtype: str = "uint8",
    ):

        try:
            normalize_factor = NORMALIZE_FACTORS[data_original_dtype]

        except KeyError:
            raise ValueError(f"Unknown {data_original_dtype=} in {NORMALIZE_FACTORS.keys()=}")

        grid_pos_gen = UniformGridPosition.build_from_volume_crop_shapes(
            volume_shape=volume_shape, 
            crop_shape=crop_shape,
            random_state=RandomState(common_random_state_seed),
        )
        
        return cls(
            volume_shape=volume_shape,
            crop_shape=crop_shape,
            
            x0y0z0_generator=grid_pos_gen,
            
            # it is too slow to use this
            et_field=ET3DConstantEverywhere.build_no_displacement(
                grid_position_generator=grid_pos_gen
            ),
            
            gt_field=GTUniformEverywhere(
                gt_type=gt_type,
                random_state=RandomState(common_random_state_seed),
                grid_position_generator=grid_pos_gen,
            ),
            
            vs_field=VSUniformEverywhere.build_plus_or_mines(
                shift=1. / normalize_factor / 2,  # half a value to both sides +/-
                grid_position_generator=grid_pos_gen,
                random_state=RandomState(common_random_state_seed),
            ),
            
            is_2halfd=is_2halfd,
        )
    
    @classmethod
    def build_setup_train01(
        cls, 
        volume_shape: Tuple[int, int, int], 
        crop_shape: Tuple[int, int, int], 
        common_random_state_seed: int,
        gt_type: Type,
        is_2halfd: bool,
        data_original_dtype: str = "uint8",
        gt_no_transpose_rot: bool = False,
    ):
        """just like build_setup_train00 with an adaptation to use GTUniformEverywhere2"""

        try:
            normalize_factor = NORMALIZE_FACTORS[data_original_dtype]

        except KeyError:
            raise ValueError(f"Unknown {data_original_dtype=} in {NORMALIZE_FACTORS.keys()=}")

        grid_pos_gen = UniformGridPosition.build_from_volume_crop_shapes(
            volume_shape=volume_shape, 
            crop_shape=crop_shape,
            random_state=RandomState(common_random_state_seed),
        )
        
        gt_build_func = GTUniformEverywhere2.gt_no_transpose_rot if gt_no_transpose_rot else GTUniformEverywhere2.build_all
        
        return cls(
            volume_shape=volume_shape,
            crop_shape=crop_shape,
            
            x0y0z0_generator=grid_pos_gen,
            
            # it is too slow to use this
            et_field=ET3DConstantEverywhere.build_no_displacement(
                grid_position_generator=grid_pos_gen
            ),
            
            gt_field=gt_build_func(
                GTUniformEverywhere2,
                gt_type=gt_type,
                random_state=RandomState(common_random_state_seed),
                grid_position_generator=grid_pos_gen,
            ),
            
            vs_field=VSUniformEverywhere.build_plus_or_mines(
                shift=1. / normalize_factor / 2,  # half a value to both sides +/-
                grid_position_generator=grid_pos_gen,
                random_state=RandomState(common_random_state_seed),
            ),
            
            is_2halfd=is_2halfd,
        )
    
    @classmethod
    def build_setup_val00(
        cls, 
        grid_pos_gen: GridPositionGenerator,
        volume_shape: Tuple[int, int, int], 
        crop_shape: Tuple[int, int, int], 
        common_random_state_seed: int,
        gt_type: Type,
        is_2halfd: bool,
    ):
        
        return cls(
            volume_shape=volume_shape,
            crop_shape=crop_shape,
            x0y0z0_generator=grid_pos_gen,
            et_field=ET3DConstantEverywhere.build_no_displacement(grid_position_generator=grid_pos_gen),
            gt_field=(
                GTConstantEverywhere.build_gt3d_identity(grid_position_generator=grid_pos_gen) if gt_type == GT3D else
                GTConstantEverywhere.build_gt2d_identity(grid_position_generator=grid_pos_gen) 
            ),
            vs_field=VSConstantEverywhere.build_no_shift(grid_position_generator=grid_pos_gen),
            is_2halfd=is_2halfd,
        )
        


@dataclass
class VolumeCropSequence(Sequence):
    """
    todo make it multi-channel enabled
    """

    data_volume: np.ndarray
    labels_volume: np.ndarray
    labels: List[int]
    meta_crop_generator: MetaCrop3DGenerator

    batch_size: int
    epoch_size: int

    meta2data: Callable[[MetaCrop3D], ndarray] = field(init=False)
    meta2labels: Callable[[MetaCrop3D], ndarray] = field(init=False)

    # each internal list corresponds to a batch
    meta_crops_hist_buffer: List[List[MetaCrop3D]] = field(init=False)
    meta_crops_hist_path: Optional[Path] = None

    output_as_2d: bool = False
    output_as_2halfd: bool = False
    use_labels_ohe: bool = False
    debug__no_data_check: InitVar[bool] = False

    @property
    def crop_shape(self):
        return self.meta_crop_generator.crop_shape

    def __post_init__(self, debug__no_data_check):

        assert not (self.output_as_2d and self.output_as_2halfd), "Choose only one or none of them."

        if self.output_as_2d:
            assert (
                self.meta_crop_generator.crop_shape[0] == 1 
                or self.meta_crop_generator.crop_shape[1] == 1 
                or self.meta_crop_generator.crop_shape[2] == 1 
            ), f"{self.meta_crop_generator.crop_shape=} not compatible with {self.output_as_2d=}, at least one axis must be of size 1."
            
        if self.output_as_2halfd:
            logger.warning(f"{self.output_as_2halfd=} only xy layers is available for 2.5d now!")
            
            assert (nlayers := self.meta_crop_generator.crop_shape[2]) % 2 == 1, f"{nlayers=} must be odd."
            assert nlayers > 1, f"{nlayers=} is not 2.5d!"
            
            assert self.meta_crop_generator.is_2halfd, "Oops, you gotta be consistent with your choices."

            # here we can assume nlayers to be odd and at least 3
            # it is supposed that the middle layer is the target
            self.target_layer_idx = (nlayers - 1) // 2

        logger.debug(f"Initializing {self.__class__.__name__}.")

        if self.data_volume.ndim > 3:
            raise NotImplementedError("Multi channel 3D not supported.")
        assert self.data_volume.ndim == 3

        assert self.data_volume.shape == self.labels_volume.shape == self.meta_crop_generator.volume_shape
        self.volume_shape = self.data_volume.shape

        if not debug__no_data_check:
            logger.debug("Checking values and labels consistency, this might be a bit slow.")
            assert (data_max_val := np.max(self.data_volume, axis=None)) <= 1., \
                f"{data_max_val=} > 1. Did you forget to normalize?"

            assert (data_min_val := np.min(self.data_volume, axis=None)) >= 0., \
                f"{data_min_val=} < 0. Did you forget to normalize?"

            assert all(lab in self.labels for lab in np.unique(self.labels_volume))

            logger.debug("Done.")

        assert self.batch_size >= 1
        assert self.epoch_size >= 1

        self.meta_crops_hist_buffer = []

        if self.meta_crops_hist_path is not None:
            if not self.meta_crops_hist_path.exists():
                logger.info("A meta crops history file path was given but it still doesn't exist. Writing csv headers.")
                with self.meta_crops_hist_path.open("w") as f:
                    f.write(f"batch_idx,{MetaCrop3D.csv_header}")
        else:
            logger.warning("No meta crops history file path given. The randomly generated crops will not be saved!")

        # noinspection PyTypeChecker,PyUnresolvedReferences
        self.meta2data = partial(
            meta2crop,
            volume=self.data_volume,
            is_label=False,
            interpolation="spline",
            spline_order=getattr(
                self.meta_crop_generator.et_field,
                "spline_order",
                None
            ),
        )
        # noinspection PyTypeChecker
        self.meta2labels = partial(
            meta2crop,
            volume=self.labels_volume,
            is_label=True,
            interpolation="nearest",
            spline_order=None,
        )

    def on_epoch_end(self):
        if self.meta_crops_hist_path is not None:
            with self.meta_crops_hist_path.open(mode='a') as f:
                for batch_idx, batch in enumerate(self.meta_crops_hist_buffer):
                    for crop_meta in batch:
                        f.write(f"{batch_idx},{crop_meta.to_csv_line()}")
        self.meta_crops_hist_buffer = []

    def __len__(self):
        """Number of batches per epoch."""
        return self.epoch_size

    def __getitem__(self, index):
        """Generate one batch of data."""
        (X, y), _ = self.__getitem_with_meta__(index)
        return X, y

    def __getitem_with_meta__(self, index):

        batch_meta_crops = self.meta_crop_generator.get(self.batch_size)
        self.meta_crops_hist_buffer.append(batch_meta_crops)

        # Initialization

        n_classes = len(self.labels)

        if self.output_as_2d:

            if not batch_meta_crops[0].is2d_on(2):
                raise NotImplementedError("todo implement support for non-xy-2d!")

            target_x_shape = (
                self.batch_size,
                self.crop_shape[0],
                self.crop_shape[1],
                1  # mono-channel for now todo make it multichannel
            )
            target_y_shape = tuple(
                [
                    self.batch_size,
                    self.crop_shape[0],
                    self.crop_shape[1],
                ] + self.use_labels_ohe * [n_classes]
            )

        elif self.output_as_2halfd:
            
            target_x_shape = (
                self.batch_size,
                self.crop_shape[0],
                self.crop_shape[1],
                self.crop_shape[2],
                # the layers are the channels here
                # todo make it *actually* multichannel
            )
            target_y_shape = tuple(
                [
                    self.batch_size,
                    self.crop_shape[0],
                    self.crop_shape[1],
                ] + self.use_labels_ohe * [n_classes]
            )

        else:  # 3d might it be (:
            target_x_shape = (
                self.batch_size,
                self.crop_shape[0],
                self.crop_shape[1],
                self.crop_shape[2],
                # 1  # mono-channel for now todo make it multichannel
            )
            target_y_shape = tuple(
                [
                    self.batch_size,
                    self.crop_shape[0],
                    self.crop_shape[1],
                    self.crop_shape[2],
                ] + self.use_labels_ohe * [n_classes]
            )

        X = np.empty(target_x_shape, dtype=np.float)
        y = np.empty(target_y_shape, dtype=np.uint8)

        for idx, meta_crop in enumerate(batch_meta_crops):

            data_crop = self.meta2data(meta_crop)
            labels_crop = self.meta2labels(meta_crop)

            # make it 2d if that's the case
            if self.output_as_2d:
                data_crop, labels_crop = (
                    (data_crop[:, :, 0], labels_crop[:, :, 0]) if meta_crop.is2d_on(2) else
                    (data_crop[:, 0, :], labels_crop[:, 0, :]) if meta_crop.is2d_on(1) else
                    (data_crop[0, :, :], labels_crop[0, :, :]) if meta_crop.is2d_on(0) else
                    (data_crop, labels_crop)
                )
                data_crop_target_shape = tuple(list(data_crop.shape) + [1])
                data_crop = data_crop.reshape(data_crop_target_shape)
                
            elif self.output_as_2halfd:
                labels_crop = labels_crop[:, :, self.target_layer_idx]
                
            else:
                pass

            if self.use_labels_ohe:
                raise NotImplementedError()  # the function here below needs to be revised
                # labels_crop = _labels_single2multi_channel(labels_crop, self.labels)

            X[idx] = data_crop  # add the channel dimension
            y[idx] = labels_crop

        return (X, y), batch_meta_crops


if __name__ == "__main__":

    n = 500
    random_probas = np.random.rand(n ** 3).reshape(n, n, n)
    random_probas /= (random_probas.sum())
    random_probas /= (random_probas.sum())
    gpg = VoxelwiseProbabilityGridPotion(
        x_range=(0, n),
        y_range=(0, n),
        z_range=(0, n),
        probabilities_volume=random_probas,
        random_state=np.random.RandomState(42),
    )
    print(gpg.get(16))
