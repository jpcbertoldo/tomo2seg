"""
3D crop generator from a volume.
todo make this module batch-enabled
todo make this all with keras backend
"""

# Standard packages
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, InitVar, replace
from enum import Enum
from functools import partial
from itertools import combinations, product
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Callable, Type, Union, ClassVar

# Installed packages
import numpy as np
from numpy import ndarray
from numpy.random import RandomState
import scipy as sp
from scipy.interpolate import RegularGridInterpolator, griddata
from scipy.ndimage import map_coordinates
from tensorflow.keras.utils import Sequence

from .logger import logger


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
    GT2D.transpose: partial(np.transpose, axes=(0, 1)),
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


def _get_random_gt_2d(random_state: RandomState) -> GT2D:
    """todo make this batch-enabled"""
    return random_state.choice(GT2D, 1, replace=False)[0]


class GT3D(Enum):
    """Canonical 3D Geometric Transformation. Any other one is equivalent to these."""
    pass


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
        assert all(range_[0] <= range_[1] for range_ in self.axes_ranges)
    
    @property
    def axes_ranges(self) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        return self.x_range, self.y_range, self.z_range

    def get(self, n: int) -> ndarray:  # (n, 3)
        assert n > 0
        grid_positions = self._concrete_getitem(n)
        assert grid_positions.shape == (n, 3)
        return grid_positions

    @abstractmethod
    def _concrete_getitem(self, n: int) -> ndarray:  # (n, 3)
        pass


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

    x_step: int
    y_step: int
    z_step: int

    def __post_init__(self):
        self.positions = np.array([
            [x, y, z]
            for z, y, x in product(
                range(*self.z_range, self.z_step),
                range(*self.y_range, self.y_step),
                range(*self.x_range, self.x_step),
            )
        ])
        logger.info(f"The {self.__class__.__name__} has {len(self.positions)=} different positions (therefore crops).")
        self.current_position = 0

    def __len__(self):
        return len(self.positions)

    def _concrete_getitem(self, n: int) -> ndarray:
        positions = self.positions[self.current_position:(new_current := self.current_position + n)]
        self.current_position = new_current if new_current < len(self.positions) else 0
        return positions


@dataclass
class ProbabilityField3D(ABC):
    """
    Each position of the volume has a probability distribution over the possible values.
    It supposes a regular grid of unitary steps in every axis.
    """
    x_range: Tuple[int, int]
    y_range: Tuple[int, int]
    z_range: Tuple[int, int]
    random_state: Optional[RandomState]  # some concrete classes are deterministic

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

    gt: Union[GT2D, GT3D]

    def _concrete_getitem(self, x: int, y: int, z: int):
        return self.gt

    @classmethod
    def build(cls, gt, **ranges):
        return cls(
            gt=gt,
            random_state=None,
            **ranges,
        )


@dataclass
class GTUniformEverywhere(ProbabilityField3D):

    gt_type: Type  # GT2D or GT3D

    def _concrete_getitem(self, x: int, y: int, z: int):
        """Every position has a uniform probability over all the possible transformations. todo make this batch-enabled"""

        if self.gt_type == GT2D:
            return _get_random_gt_2d(random_state=self.random_state)

        elif self.gt_type == GT3D:
            raise NotImplementedError("GT3D not implemented yet.")

        else:
            raise ValueError(f"Unknown type of Geometric Transformation. {self.gt_type=}")


@dataclass
class VSConstantEverywhere(ProbabilityField3D):
    """Values shift is always the same everywhere."""

    shift: float  # shift on the normalized range [0, 1]

    def _concrete_getitem(self, x: int, y: int, z: int):
        return self.shift

    @classmethod
    def build(cls, shift, **ranges):
        return cls(
            shift=shift,
            random_state=None,
            **ranges,
        )


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
class ET3DConstantDisplacementEverywhere(ProbabilityField3D):

    displacement: Optional[ET]

    def _concrete_getitem(self, x: int, y: int, z: int):
        return self.displacement

    @classmethod
    def build(cls, displacement, **ranges):
        return cls(
            displacement=displacement,
            random_state=None,
            **ranges,
        )


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

    elastic_param: InitVar[Union[float, Tuple[float, float, float]]]

    crop_shape: Tuple[int, int, int]
    cuboid_shape: Tuple[float, float, float] = field(init=False)

    # the limits of the data grid, i.e. the elastic crop cannot go outside of this range
    crop_xlim: Tuple[int, int]
    crop_ylim: Tuple[int, int]
    crop_zlim: Tuple[int, int]

    spline_order: int = 3  # default copied from etienne's code

    def __post_init__(self, elastic_param):
        if isinstance(elastic_param, float):
            # cuboids proportional to the crop's size on each axis
            self.cuboid_shape = (
                elastic_param * self.crop_shape[0],
                elastic_param * self.crop_shape[1],
                elastic_param * self.crop_shape[2],
            )
        elif isinstance(elastic_param, tuple):
            # cuboid shape given
            self.cuboid_shape = elastic_param
        else:
            raise ValueError()

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

        return ET(**displacements)

    @classmethod
    def build_half_voxel_cuboid(cls, crop_shape, crop_xlim, crop_ylim, crop_zlim):
        """
        Each voxel's probability cuboid is of the size of a voxel (half to each direction)
        in all the axes, so there is no overlap between each voxel's domain.
        """
        return cls(
            elastic_param=(.5, .5, .5),
            crop_shape=crop_shape,
            crop_xlim=crop_xlim,
            crop_ylim=crop_ylim,
            crop_zlim=crop_zlim,
        )


@dataclass
class MetaCrop3D:
    csv_header: ClassVar[str] = "x,y,z,elastic_transformation,geometric_transformation,value_shift"

    x: slice
    y: slice
    z: slice
    et: Optional[ET]
    gt: Union[GT2D, GT3D]
    vs: float  # shift on the normalized range [0, 1]

    @property
    def slice(self) -> Tuple[slice, slice, slice]:
        return self.x, self.y, self.z

    @property
    def shape(self) -> Tuple[int, int, int]:
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
        assert x_ in (0, 1)
        assert y_ in (0, 1)
        assert z_ in (0, 1)
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
    def is2d(self):
        return any(self.is2d_on(axis_idx) for axis_idx in range(3))

    def to_csv_line(self) -> str:
        return f"{self.x},{self.y},{self.z},{repr(self.et)},{self.gt.value},{self.vs}"


def _labels_single2multi_channel(im: ndarray, labels: List[int]) -> np.ndarray:
    """ Extract labels of interest from input segmentation """
    assert im.ndim == 3
    im_out = np.zeros((im.shape[0], im.shape[1], im.shape[2], len(labels)))
    for final_label, label in enumerate(labels):
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
    assert (is_label and interpolation == "nearest") or not is_label, "Labels should only be interpolated witht he nearest neighbor because it is a categorical value."
    assert not (meta_crop.et is not None and interpolation == "spline" and spline_order is None), "If elastic transformation is used then the interpolation spline order has to be given."

    if meta_crop.et is None:
        crop = volume[meta_crop.slice].copy()
    else:
        corners_binaries = list(product((0, 1), (0, 1), (0, 1)))

        ijk_grid_corners = np.array([
            [
                0 if ci_ == 0 else meta_crop.shape[0] - 1,
                0 if cj_ == 0 else meta_crop.shape[1] - 1,
                0 if ck_ == 0 else meta_crop.shape[2] - 1,
            ]
            for ci_, cj_, ck_ in corners_binaries
        ])  # 8 x 3

        # n_grid_points = (meta_crop.shape[0] * meta_crop.shape[1] * meta_crop.shape[2])
        ijk_grid_points = np.array(list(product(
            np.arange(meta_crop.shape[0]),
            np.arange(meta_crop.shape[1]),
            np.arange(meta_crop.shape[2]),
        )))

        et_grid_corner_coords = np.array([
            meta_crop.get_et_corner_coords(ci_, cj_, ck_)
            for ci_, cj_, ck_ in corners_binaries
        ])  # 8 x 3

        def get_et_grid_coords(axis):
            return sp.interpolate.griddata(
                points=ijk_grid_corners,
                values=et_grid_corner_coords[:, axis].ravel().copy(),
                method="linear",
                xi=ijk_grid_points,
            )  # n_grid_points x 1

        et_grid_coords = np.stack(
            [get_et_grid_coords(dim) for dim in range(3)], axis=-1
        )  # n_grid_points x 3

        if interpolation == "spline":
            # todo test if t will be too slow cuz it might do a spline for the entire volume (?)
            crop = map_coordinates(
                input=volume,
                coordinates=et_grid_coords,
                # todo check how this works, can it break it order=3 and the crop is in the corner?
                # todo visualize the effect with crops at the borders because this will search for 3rd order derivatives
                order=spline_order,
                mode="mirror",
            )
        elif interpolation == "nearest":
            # todo keep this somewhere to avoid generating?
            crop = sp.interpolate.interpn(
                points=(
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
    if meta_crop.is2d:
        func = _GT2D_VAL2FUNC[meta_crop.gt.value]

        if meta_crop.is2d_on(0):
            sx = crop.shape[1]
            sy = crop.shape[2]
            crop = func(crop.reshape(sx, sy)).reshape(1, sx, sy)
        elif meta_crop.is2d_on(1):
            sx = crop.shape[0]
            sy = crop.shape[2]
            crop = func(crop.reshape(sx, sy)).reshape(sx, 1, sy)
        else:
            sx = crop.shape[0]
            sy = crop.shape[1]
            crop = func(crop.reshape(sx, sy)).reshape(sx, sy, 1)
    else:
        pass

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
    elastic_transformation_field: ProbabilityField3D
    geometric_transformation_field: ProbabilityField3D
    value_shift_field: ProbabilityField3D

    def __post_init__(self):

        assert all(0 < s <= axis_size for s, axis_size in zip(self.crop_shape, self.volume_shape))

        # make sure it wont try to slice something beyond the 3d array
        # the crops are referenced by their 000 corner (i.e. the lowest x, y, z)
        for axis_idx in range(3):
            axis_range = self.x0y0z0_generator.axes_ranges[axis_idx]
            volume_axis_size = self.volume_shape[axis_idx]
            crop_axis_size = self.crop_shape[axis_idx]
            assert (
                    0 <= axis_range[0] and axis_range[1] <= volume_axis_size - crop_axis_size
            ), f"{axis_range=} of {axis_idx=} is incompatible with {volume_axis_size=} and { crop_axis_size=}"

        for axis_idx in range(3):
            xyz_range = self.x0y0z0_generator.axes_ranges[axis_idx]
            et_range = self.elastic_transformation_field.axes_ranges[axis_idx]
            gt_range = self.geometric_transformation_field.axes_ranges[axis_idx]
            vs_range = self.value_shift_field.axes_ranges[axis_idx]
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
                elastic_transformation=self.elastic_transformation_field[x, y, z],
                geometric_transformation=self.geometric_transformation_field[x, y, z],
                value_shift=self.value_shift_field[x, y, z],
            )
            for x, y, z in x0y0z0_array
        ]


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

    use_labels_ohe: bool = False
    debug__no_data_check: InitVar[bool] = False

    @property
    def crop_shape(self):
        return self.meta_crop_generator.crop_shape

    def __post_init__(self, debug__no_data_check):

        logger.debug(f"Initializing {self.__class__.__name__}.")

        if self.data_volume.ndim > 3:
            raise NotImplementedError("Multi channel 3D not supported.")
        assert self.data_volume.ndim == 3

        assert self.data_volume.shape == self.labels_volume.shape == self.meta_crop_generator.volume_shape
        self.volume_shape = self.data_volume.shape

        if not debug__no_data_check:
            logger.debug("Checking values and labels consistency, this might be a bit slow.")
            assert (data_max_val := np.max(self.data_volume, axis=None)) <= 1., f"{data_max_val=} > 1. Did you forget to normalize?"
            assert (data_min_val := np.min(self.data_volume, axis=None)) >= 0., f"{data_min_val=} < 0. Did you forget to normalize?"
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
                self.meta_crop_generator.elastic_transformation_field,
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

        batch_meta_crops = self.meta_crop_generator.get(self.batch_size)
        self.meta_crops_hist_buffer.append(batch_meta_crops)

        # Initialization
        X = np.empty(
            (
                self.batch_size,
                self.crop_shape[0],
                self.crop_shape[1],
                self.crop_shape[2],
                # mono-channel for now todo make it multichannel
            ),
            dtype=np.float
        )
        y = np.empty(
            (
                self.batch_size,
                self.crop_shape[0],
                self.crop_shape[1],
                self.crop_shape[2],
                len(self.labels) if self.use_labels_ohe else 1,
            ),
            dtype=np.uint8
        )

        for idx, meta_crop in enumerate(batch_meta_crops):
            data_crop = self.meta2data(meta_crop)
            labels_crop = self.meta2labels(meta_crop)
            if self.use_labels_ohe:
                labels_crop = _labels_single2multi_channel(labels_crop, self.labels)
            else:
                labels_crop = labels_crop.reshape(*labels_crop.shape, 1)
            X[idx], y[idx] = data_crop, labels_crop
        return X, y

    def get_clone(self, new_batch_size, new_epoch_size) -> "VolumeCropSequence":
        raise NotImplementedError("todo reconstruct the random states here")
        # noinspection PyArgumentList
        return replace(self, **dict(batch_size=new_batch_size, epoch_size=new_epoch_size))
