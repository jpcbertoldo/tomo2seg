"""
Keras 3D crop generator from a volume.
"""

# Standard packages
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import List, Optional, Tuple, Dict, Callable

# Installed packages
import numpy as np
from numpy import ndarray
from numpy.random import RandomState
from tensorflow.keras.utils import Sequence


# GT = Geometric Transformation
class GT2D(Enum):
    """"""
    identity = 0
    rotation90 = 1
    flip_horizontal = 2
    flip_vertical = 3
    transpose = 4
    flip_horizontal__transpose = 5
    flip_vertical__flip_horizontal = 6
    rotation90__flip_vertical = 7


# a dictionary of transformations so that they can be referenced with a string key
GT2D_FUNCTIONS: Dict[GT2D, Callable[[ndarray], ndarray]] = {
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
GT2D_FUNCTIONS.update({
    GT2D.flip_horizontal__transpose: _compose(
        f=GT2D_FUNCTIONS[GT2D.flip_horizontal],
        g=GT2D_FUNCTIONS[GT2D.transpose]
    ),
    GT2D.flip_vertical__flip_horizontal: _compose(
        f=GT2D_FUNCTIONS[GT2D.flip_vertical],
        g=GT2D_FUNCTIONS[GT2D.flip_horizontal]
    ),
    GT2D.rotation90__flip_vertical: _compose(
        f=GT2D_FUNCTIONS[GT2D.rotation90],
        g=GT2D_FUNCTIONS[GT2D.flip_vertical]
    ),
})

GT2D_VAL2FUNC = {
    gt.value: func for gt, func in GT2D_FUNCTIONS.items()
}


@dataclass
class MetaCrop3D:

    

_SliceParams = namedtuple("_SliceParams", [
    "axis",  # the axis orthogonal to the  slice plane (0: x, 1: y, 2: z),
    "coordinate",  # the offset of the slice on the axis
    # ex: if axis = 2 then the slice is on the plane XY
    "corner",
    "geometric_transformation",
])


def _get_random_transformations(n: int, random_state: RandomState) -> List[GT]:
    """
    :param n: number of random transformations to select
    :return:
        `n` (randomly selected) dictionary keys from `GEOM_TRANSF`
    """
    if n == 0:
        return [GT.identity]

    all_geom_transf_names = list(GEOM_TRANSF.keys())

    if n >= len(GEOM_TRANSF):
        return all_geom_transf_names

    return random_state.choice(all_geom_transf_names, n, replace=False)


def _labels_single2multi_channel(im: ndarray, labels: List[int]) -> np.ndarray:
    """ Extract labels of interest from input segmentation """
    im_out = np.zeros((im.shape[0], im.shape[1], len(labels)))
    for final_label, label in enumerate(labels):
        im_out[:, :, final_label] = (im == label).astype(np.uint8)
    return im_out


def _generate_one_slice(
        source_volume: ndarray,
        label_volume: ndarray,
        crop_size: int,
        labels: List[int],
        slice_param: _SliceParams,
) -> (np.ndarray, np.ndarray):
    # todo return pixel-wise weights as well

    # 1: get the entire slice
    if slice_param.axis == 0:
        data_im = source_volume[slice_param.coordinate, :, :]
        labels_im = label_volume[slice_param.coordinate, :, :]

    elif slice_param.axis == 1:
        data_im = source_volume[:, slice_param.coordinate, :]
        labels_im = label_volume[:, slice_param.coordinate, :]

    elif slice_param.axis == 2:
        data_im = source_volume[:, :, slice_param.coordinate]
        labels_im = label_volume[:, :, slice_param.coordinate]

    else:
        raise ValueError(f"slice_param.axis={slice_param.axis} unknown value. It must be in {{0, 1, 2}}.")

    # 2: crop it from the right corner
    x_start, x_end = (
        (0, crop_size)
        if slice_param.corner in (_Corner.BL, _Corner.UL) else
        (- crop_size, data_im.shape[0])
    )
    y_start, y_end = (
        (0, crop_size)
        if slice_param.corner in (_Corner.UL, _Corner.UR) else
        (-crop_size, data_im.shape[1])
    )

    data_im = np.copy(data_im[x_start:x_end, y_start:y_end])
    labels_im = np.copy(labels_im[x_start:x_end, y_start:y_end])

    # 3: apply augmentation
    transformation = GEOM_TRANSF[slice_param.geometric_transformation.value]
    data_im = transformation(data_im)
    labels_im = transformation(labels_im)

    # 4: labels to 1-hot encode
    labels_im = _labels_single2multi_channel(labels_im, labels)

    return data_im, labels_im


def _generate_slices(
        source_volume: ndarray,
        label_volume: ndarray,
        crop_size: int,
        labels: List[int],
        slice_params: List[_SliceParams]
) -> (np.ndarray, np.ndarray):
    """ Generates image slices from their _SliceParams objects """

    n_slices = len(slice_params)
    n_labels = len(labels)

    # Initialization
    X = np.empty(
        (n_slices, crop_size, crop_size, 1), dtype=np.float
    )
    y = np.empty(
        (n_slices, crop_size, crop_size, n_labels), dtype=np.uint8
    )

    for example_index, sp in enumerate(slice_params):
        X[example_index, :, :, 0], y[example_index] = _generate_one_slice(
            source_volume, label_volume, crop_size, labels, sp
        )

    return X, y


class VolumeImgSegmSequence(Sequence):
    """
    todo implement elastic augmentation
    todo implement value shift augmentation
    """

    def __init__(
        self,
        source_volume: np.ndarray,
        label_volume: np.ndarray,
        labels: List[int] = None,
        axes: Tuple[int] = (2,),
        batch_size: int = 1,
        shuffle: bool = True,
        normalization_const: int = 255,
        n_geometric_augmentations: int = 0,
        random_state: int = 42,
        crop_size: int = 224,
        force_shorter_epoch: Optional[int] = None  # todo assert, doc, and add a warning for this
    ):
        """
        Args:
            source_volume: 3D np.ndarray volume where the data slices come from
            label_volume: 3D np.ndarray volume where the respective label slices come from
            labels: iterable of segmentation labels to be considered. Other labels are ignored.
            axes:
            batch_size: batch size (default: 1).
            shuffle: should we shuffle the images after each epoch? (default: True).
            normalization_const:
            n_geometric_augmentations: number of random geometric augmentations per image.
            random_state:
        Implementation based on:
        https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
        """
        assert 1 <= len(axes) <= 3
        assert all(0 <= p <= 2 for p in axes)
        assert isinstance(random_state, int)
        assert all(crop_size <= dim for dim in source_volume.shape)
        assert source_volume.shape == label_volume.shape
        assert (0 <= n_geometric_augmentations) or (n_geometric_augmentations == -1)

        self.source_volume = source_volume
        self.label_volume = label_volume
        self.labels = labels
        self.axes = axes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.normalization_const = normalization_const
        self.n_geometric_augmentations = len(GEOM_TRANSF) if n_geometric_augmentations == -1 else min(n_geometric_augmentations, len(GEOM_TRANSF))
        self.random_state: RandomState = RandomState(seed=random_state)
        self.crop_size = crop_size
        self.force_shorter_epoch = force_shorter_epoch

        self.shape = self.source_volume.shape

        # noinspection PyTypeChecker
        self.slice_params: Dict[int, _SliceParams] = dict(enumerate([
            _SliceParams(ax, coord, corn, geo_transf)
            for ax in self.axes
            for coord in range(self.shape[ax])
            for corn in list(_Corner)
            # select a number of random geometric augmentations, or None, or all
            for geo_transf in _get_random_transformations(
                n=self.n_geometric_augmentations, random_state=self.random_state
            )
        ]))
        self.ids: List[int] = list(self.slice_params.keys())

        self.on_epoch_end()  # shuffle (or not)

    def on_epoch_end(self):
        """Shuffle ids - if necessary - at the end of each epoch"""
        if self.shuffle is True:
            self.random_state.shuffle(self.ids)

    def __len__(self):
        """Number of batches per epoch"""
        if self.force_shorter_epoch is not None:
            return self.force_shorter_epoch
        return int(np.floor(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        batch_ids = self.ids[index * self.batch_size : (index + 1) * self.batch_size]
        batch_slice_params = [self.slice_params[id_] for id_ in batch_ids]
        X, y = _generate_slices(
            source_volume=self.source_volume,
            label_volume=self.label_volume,
            crop_size=self.crop_size,
            labels=self.labels,
            slice_params=batch_slice_params
        )
        return X / self.normalization_const, y


