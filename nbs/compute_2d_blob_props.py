# coding: utf-8
"""
this is in a script because it uses multiprocessing and jupyter does not deal well
with it, making the workers hang indefinitely
"""

import functools
import logging
from pathlib import Path

from pymicro.file import file_utils
import numpy as np
from numpy import ndarray
from progressbar import progressbar
import skimage
from skimage import measure
from typing import Optional, List, Dict
import multiprocessing
import pandas as pd

from tomo2seg.logger import logger


PROPS = [
    "label", 
    "area", "bbox", "bbox_area", "centroid", "eccentricity", "euler_number", "extent",
    "filled_area", "inertia_tensor_eigvals", "local_centroid", "major_axis_length", "minor_axis_length",
    "perimeter", "solidity", 
    # todo add properties using the intensity image
]


_get_blob_props_func = functools.partial(
    skimage.measure.regionprops_table,
    cache=True,
    separator="-",
    properties=PROPS
)


def _do_instances_slice(args):
    idx, instances = args
    return {**_get_blob_props_func(instances), **{"slice_idx": idx}}


def get_2d_blobs_from_slices(slices_array: ndarray, label_to_search: int, n_processes: Optional[int]) -> dict:

    if (slices_type := type(slices_array)) == np.ndarray:
        instances_slices = (n_slices := slices_array.shape[0]) * [None]
    elif slices_type == list:
        instances_slices = (n_slices := len(slices_array)) * [None]
    else:
        raise ValueError("Unknown type of `slices_array`")

    logger.debug("separating connected blobs")
    for idx, slice_ in progressbar(
        enumerate(slices_array),
        max_value=n_slices,
        prefix="slices ",
    ):
        instances_slices[idx] = skimage.measure.label(
            slice_ == label_to_search, 
            connectivity=2, background=0, return_num=False
        )
    
    logger.debug("computing blobs' properties")
    with multiprocessing.Pool(n_processes) as p:
        mapresult = p.map_async(
            _do_instances_slice,  # get the properties of all blobs in a slice
            enumerate(instances_slices)
        )
        blobs_per_slice = mapresult.get()
    
    logger.debug("properties' types and adding `slice_idx`")
    for blobs in blobs_per_slice:
        blobs['label'] = label_to_search * np.ones_like(blobs['label'])
        blobs['slice_idx'] = blobs['slice_idx'] * np.ones_like(blobs['label'])
        for k, v in blobs.items():
            if v.dtype in (np.float, np.float64, np.float32, np.float128):
                blobs[k] = v.astype(np.float16)  # reduce memory usage
        
    prop_keys = list(blobs_per_slice[0].keys())

    logger.debug("concatenating all blobs props")
    return {
        key: np.concatenate([
            blobs[key] for blobs in blobs_per_slice
        ])
        for key in prop_keys
    }


get_2d_blobs_from_slices_only_porosity = functools.partial(
    get_2d_blobs_from_slices,
    label_to_search=2,  # porosity
    n_processes=None,  # use all 
)

get_2d_blobs_from_slices_only_fiber = functools.partial(
    get_2d_blobs_from_slices,
    label_to_search=1,  # fiber
    n_processes=None,  # use all
)


def compute_all_axes(labels_volume, x_blobs_path, y_blobs_path, z_blobs_path):
    logger.info("Processing z-slices")
    if z_blobs_path is None:
        logger.warning("skipping...")

    else:
        logger.debug("Preparing slices")
        axis_slices = [
            labels_volume[:, :, idx].copy()
            for idx in range(labels_volume.shape[2])
        ]
        logger.debug("Computing blobs")
        blobs2d_porosity = pd.DataFrame(get_2d_blobs_from_slices_only_porosity(axis_slices))
        logger.debug("Saving")
        blobs2d_porosity.to_csv(z_blobs_path)

    logger.info("Processing y-slices")
    if y_blobs_path is None:
        logger.warning("skipping...")

    else:
        logger.debug("Preparing slices")
        axis_slices = [
            labels_volume[:, idx, :].copy()
            for idx in range(labels_volume.shape[1])
        ]
        logger.debug("Computing blobs")
        blobs2d_porosity = pd.DataFrame(get_2d_blobs_from_slices_only_porosity(axis_slices))
        logger.debug("Saving")
        blobs2d_porosity.to_csv(y_blobs_path)

    logger.info("Processing x-slices")
    if x_blobs_path is None:
        logger.warning("skipping...")

    else:
        logger.debug("Preparing slices")
        axis_slices = [
            labels_volume[idx, :, :].copy()
            for idx in range(labels_volume.shape[0])
        ]

        logger.debug("Computing blobs")
        blobs2d_porosity = pd.DataFrame(get_2d_blobs_from_slices_only_porosity(axis_slices))
        logger.debug(f"{blobs2d_porosity.shape=}")

        logger.debug("Saving")
        blobs2d_porosity.to_csv(x_blobs_path)


def main_porosity_all_axes(volume_name: str, volume_version: str, labels_version: str = None):

    from tomo2seg.data import Volume

    volume = Volume.with_check(volume_name, volume_version)
    logger.info(f"{volume=}")

    hst_read = lambda x: functools.partial(
        # from pymicro
        file_utils.HST_read,
        # pre-loaded kwargs
        autoparse_filename=False,  # the file names are not properly formatted
        data_type=volume.metadata.dtype,
        dims=volume.metadata.dimensions,
        verbose=True,
    )(str(x))  # it doesn't accept paths...

    logger.info("Loading data from disk.")

    if labels_version is not None:
        logger.info(
            f"*Input* versioned labels. {labels_version=}.\n"
            f"{(labels_path := volume.versioned_labels_path(labels_version))=}"
        )
    else:
        logger.info(
            f"*Input* labels (no version)\n"
            f"{(labels_path := volume.labels_path)=}"
        )

    labels_volume = hst_read(labels_path)
    logger.debug(f"{labels_volume.shape=}")

    logger.debug(f"{(blobs_prefix_path := str(labels_path)[:-4])=}")

    # [filenames]
    blobs_paths: Dict[str, Path] = {
        "x_blobs_path": blobs_prefix_path + ".porosity.x-blobs.csv",
        "y_blobs_path": blobs_prefix_path + ".porosity.y-blobs.csv",
        "z_blobs_path": blobs_prefix_path + ".porosity.z-blobs.csv",
    }

    for kwarg, path in blobs_paths.items():
        path = Path(path)
        logger.debug(f"{kwarg}={path}")
        if path.exists():
            logger.warning(f"File {path} already exists. It will be skipped.")
            blobs_paths[kwarg] = None

    logger.info("Start computation!")
    compute_all_axes(
        labels_volume,
        **blobs_paths
    )
    logger.info("End!")


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    logger.debug(f"list of properties extracted from 2d blobs: {PROPS}")

    from tomo2seg.data import VOLUME_PRECIPITATES_V1 as VOL_NAME_VERSION
    logger.debug(f"{VOL_NAME_VERSION=}  {(labels_version := 'original')=}")

    main_porosity_all_axes(
        *VOL_NAME_VERSION, labels_version=labels_version
    )

