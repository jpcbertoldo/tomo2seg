#!/usr/bin/env python
# coding: utf-8

from functools import partial
import logging
import pathlib
from pathlib import Path
from pprint import pprint
import sys
from typing import *
import time
import yaml
from yaml import YAMLObject
import copy
import functools
import itertools
import os

import humanize
from matplotlib import pyplot as plt, cm
import numpy as np
from numpy import ndarray
import pandas as pd
from pymicro.file import file_utils
import tensorflow as tf
from numpy.random import RandomState
from progressbar import progressbar as pbar
from enum import Enum
import re
from enum import Enum
from matplotlib import patches

from tensorflow import keras
from tensorflow.keras import utils
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras import losses
from tensorflow.keras import layers

from cnn_segm import keras_custom_loss

from tomo2seg import modular_unet
from tomo2seg.logger import logger
from tomo2seg import data, viz
from tomo2seg.data import Volume
from tomo2seg.metadata import Metadata
from tomo2seg.volume_sequence import (
    VolumeCropSequence, MetaCrop3DGenerator, VSConstantEverywhere, 
    GTConstantEverywhere, SequentialGridPosition, ET3DConstantEverywhere
)
from tomo2seg import volume_sequence
from tomo2seg.model import Model as Tomo2SegModel
from tomo2seg.data import EstimationVolume
from tomo2seg import AggregationStrategy
from tomo2seg import viz


# # Setup

# In[4]:


logger.setLevel(logging.DEBUG)


# In[5]:


random_state = 42
random_state = np.random.RandomState(random_state)
runid = int(time.time())
logger.info(f"{runid=}")


# In[6]:


logger.debug(f"{tf.__version__=}")
logger.info(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}\nThis should be 2 on R790-TOMO.")
logger.debug(f"Both here should return 2 devices...\n{tf.config.list_physical_devices('GPU')=}\n{tf.config.list_logical_devices('GPU')=}")

# xla auto-clustering optimization (see: https://www.tensorflow.org/xla#auto-clustering)
# this seems to break the training
tf.config.optimizer.set_jit(False)

# get a distribution strategy to use both gpus (see https://www.tensorflow.org/guide/distributed_training)
strategy = tf.distribute.MirroredStrategy()  
logger.debug(f"{strategy=}")


# # Options

# In[7]:


# this will later be useful when i transform this in python script
save_probas_by_class = True

debug__save_figs = True
debug__materialize_crops = False
debug__save_processed_crops = False
probabilities_dtype = np.float16


# todo integrate this to the model object instead
class ModelType(Enum):
    input2d = 0
    input2halfd = 1
    input3d = 2


# In[9]:


tomo2seg_model = Tomo2SegModel.build_from_model_name(
    "unet3d.vanilla03-f08.fold000.1606-842-005"
)
logger.info(f"{tomo2seg_model=}")

model_type = ModelType.input3d
logger.info(f"{model_type.name=}")


# In[10]:


with strategy.scope():
    model = tf.keras.models.load_model(
        tomo2seg_model.autosaved_model_path_str,
        compile=False
    )
    
    in_ = model.layers[0]
    in_shape = in_.input_shape[0]
    input_n_channels = in_shape[-1:]

    logger.debug(f"{input_n_channels=}")
    
    # make it capable of getting any dimension in the input
    anysize_input = layers.Input(
        shape=[None, None, None] + list(input_n_channels),
        name="input_any_image_size"
    )
    
    logger.debug(f"{anysize_input=}")
    
    model.layers[0] = anysize_input
    
    # todo keep this somewhere instead of copying and pasting
    optimizer = optimizers.Adam()
    loss_func = keras_custom_loss.jaccard2_loss

    model.compile(loss=loss_func, optimizer=optimizer)


# # Data

# In[11]:


from tomo2seg.datasets import (
    VOLUME_COMPOSITE_V1 as VOLUME_NAME_VERSION,
#     VOLUME_COMPOSITE_V1_REDUCED as VOLUME_NAME_VERSION,
    VOLUME_COMPOSITE_V1_LABELS_REFINED3 as LABELS_VERSION
)

volume_name, volume_version = VOLUME_NAME_VERSION
labels_version = LABELS_VERSION

logger.info(f"{volume_name=}")
logger.info(f"{volume_version=}")
logger.info(f"{labels_version=}")


# In[ ]:


# Metadata/paths objects

## Volume
volume = Volume.with_check(
    name=volume_name, version=volume_version
)
logger.info(f"{volume=}")

def _read_raw(path_: Path, volume_: Volume): 
    # from pymicro
    return file_utils.HST_read(
        str(path_),  # it doesn't accept paths...
        # pre-loaded kwargs
        autoparse_filename=False,  # the file names are not properly formatted
        data_type=volume.metadata.dtype,
        dims=volume.metadata.dimensions,
        verbose=True,
    )

read_raw = partial(_read_raw, volume_=volume)

logger.info("Loading data from disk.")

## Data
voldata = read_raw(volume.data_path) / 255  # normalize
logger.debug(f"{voldata.shape=}")

# partition = volume.train_partition
# partition = volume.val_partition
partition = volume.test_partition

data_volume = partition.get_volume_partition(voldata)

del voldata

agg_strategy = AggregationStrategy.average_probabilities

logger.debug(f"{data_volume.shape=} {partition=} {agg_strategy=} {runid=}")


# # Estimation Volume

# In[ ]:


estimation_volume = EstimationVolume.from_objects(
    volume=volume, 
    model=tomo2seg_model, 
    set_partition=partition,
    runid=runid,
)
estimation_volume["aggregation_strategy"] = agg_strategy.name

logger.info(f"{estimation_volume=}")


# # Processing

# In[ ]:


if debug__save_figs:
    figs_dir = estimation_volume.dir
    logger.debug(f"{figs_dir=}")
    figs_dir.mkdir(exist_ok=True)


# # Shapes and steps 

# In[ ]:


DEFAULT_3D_DIM = 48

volume_shape = data_volume.shape
logger.debug(f"{volume_shape=}")

# it has to be multiple of 16 because of the 4 cascaded 2x2-strided 2x2-downsamplings in u-net
if model_type == ModelType.input2d:
    dims_multiple_16 = [int(16 * np.floor(dim / 16)) for dim in partition.shape[:2]]
    crop_shape = tuple(dims_multiple_16 + [1])  # x-axis, y-axis, z-axis

elif model_type == ModelType.input2halfd:
    raise NotImplemented()
    
elif model_type == ModelType.input3d:
    dims_multiple_16 = [
        int(16 * np.floor(dim / 16)) for dim in partition.shape
    ]
    nvoxels_per_crop = dims_multiple_16[0] * dims_multiple_16[1] * dims_multiple_16[2]
    
    if nvoxels_per_crop > DEFAULT_3D_DIM ** 3:
        logger.warning(f"If {dims_multiple_16=} ==> {nvoxels_per_crop=}, which is too big. Using default dimension {DEFAULT_3D_DIM=}**3.")
        dims_multiple_16 = [DEFAULT_3D_DIM, DEFAULT_3D_DIM, DEFAULT_3D_DIM]
    crop_shape = tuple(dims_multiple_16)  # x-axis, y-axis, z-axis

logger.debug(f"{dims_multiple_16=}")
logger.debug(f"{crop_shape=}")

n_steps = tuple(
    int(np.ceil(vol_dim / crop_dim))
    for vol_dim, crop_dim in zip(volume_shape, crop_shape)
)
logger.debug(f"{n_steps=}")

def get_coordinates_iterator(n_steps_):
    assert len(n_steps_) == 3
    return itertools.product(*(range(n_steps_[dim]) for dim in range(3)))

get_ijk_iterator = functools.partial(
    get_coordinates_iterator, copy.copy(n_steps)
)

get_kji_iterator = functools.partial(
    get_coordinates_iterator, tuple(reversed(n_steps))
)

# coordinates (xs, ys, and zs) of the front upper left corners of the crops
x0s, y0s, z0s = tuple(
    tuple(map(
        int, 
        np.linspace(0, vol_dim - crop_dim, n)
    ))
    for vol_dim, crop_dim, n in zip(volume_shape, crop_shape, n_steps)
)
logger.debug(f"""{min(x0s)=}, {max(x0s)=}, {len(x0s)=}
{min(y0s)=}, {max(y0s)=}, {len(y0s)=}
{min(z0s)=}, {max(z0s)=}, {len(z0s)=}
""")


# # Orthogonal slices figs

# In[ ]:


if debug__save_figs:
    
    fig, axs = plt.subplots(2, 2, figsize=(sz := 15, sz), dpi=120)
    fig.set_tight_layout(True)
    
    display = viz.OrthogonalSlicesDisplay(
        volume=data_volume,
        volume_name=volume.fullname,
    ).plot(axs=axs,)
    
    logger.info(f"Saving figure {(figname := display.title + '.png')=}")
    display.fig_.savefig(
        fname=figs_dir / figname,
        dpi=200, format="png",
        metadata=display.metadata,
    )    


# # Crops coordinates 

# In[ ]:


logger.debug("Generating the crop coordinates.")

crops_coordinates = np.array(
    [
        (
            (x0, x0 + crop_shape[0]), 
            (y0, y0 + crop_shape[1]),
            (z0, z0 + crop_shape[2]),
        )
        for x0, y0, z0 in itertools.product(x0s, y0s, z0s)
    ], 
    dtype=tuple
).reshape(len(x0s), len(y0s), len(z0s), 3, 2).astype(int)  # 3 = nb of dimenstions, 2 = (start, end)

logger.debug(f"{crops_coordinates.shape=}\n{crops_coordinates[0, 0, 0]=} ")

# 'F' reshapes with x varying fastest and z slowest
crops_coordinates_sequential = crops_coordinates.reshape(-1, 3, 2, order='F')  

logger.debug(f"{crops_coordinates_sequential.shape=}\n{crops_coordinates_sequential[0]=} ")


# # Crops (if `debug__materialize_crops`)

# In[ ]:


if debug__materialize_crops:
    logger.info("Materializing crops")
    
    crops_sequential = np.array([
        data_volume[tuple(slice(*coords_) for coords_ in coords)]
        for coords in pbar(crops_coordinates_sequential, max_value=crops_coordinates_sequential.shape[0])
    ])
    logger.debug(f"{crops_sequential.shape=}")

    crops_target_shape = list(crops_coordinates.shape[:3]) + list(crop_shape)
    logger.debug(f"{crops_target_shape=}")

    # 'F' reshapes with x varying fastest and z slowest
    # this option is necessary because `crops_coordinates` was reshaped with it
    crops = crops_sequential.reshape(crops_target_shape, order="F")
    del crops_sequential
    logger.debug(f"{crops.shape=}")
    
    if debug__save_processed_crops:
        fname = estimation_volume.debug__crops_coordinates_path
        logger.info(f"Saving crops coordinates at {fname=}")
        np.save(fname, crops_coordinates)
        
        fname = estimation_volume.debug__crops_path
        logger.info(f"Saving materialized crops at {fname=}")
        np.save(fname, crops)
        
    if debug__save_figs:

        n_crop_plots = 3
        logger.debug(f"Plotinng {n_crop_plots=} examples of 3d crops.")

        for n, (k, j, i) in enumerate(get_kji_iterator()):

            if n >= n_crop_plots:
                break

            ijk = (i, j, k)
            one_crop = crops[i, j, k]
            logger.debug(f"{ijk=} {one_crop.shape=}")

            fig, axs = plt.subplots(
                nrows=2, ncols=2,
                figsize=(sz := 20, sz), 
                dpi=120,
                gridspec_kw={"wspace": (gridspace := .01), "hspace": .5 * gridspace}
            )

            display = viz.OrthogonalSlicesDisplay(
                volume=one_crop,
                volume_name=volume.fullname + f".debug.crop-{ijk=}",
            ).plot(axs=axs, with_cuts=False)

            logger.info(f"Saving figure {(figname := display.title + '.png')=}")
            display.fig_.savefig(
                fname=figs_dir / figname,
                format="png",
                metadata=display.metadata,
            )       
            plt.close()


# # Segment an example

# In[ ]:


crop_ijk = (0, 0, 0)
i, j, k = crop_ijk
crop_coords = crops_coordinates[i, j, k]
logger.info(f"Segmenting one crop for debug {crop_ijk=} {crop_coords=}")

if debug__materialize_crops:
    crop_data = crops[i, j, k]
else:
    slice3d = tuple(slice(*coords_) for coords_ in crop_coords)
    crop_data = data_volume[slice3d]
    del slice3d
    
logger.debug(f"{crop_data.shape=}")


# In[ ]:


# [model] - i call it with a first crop bc if something goes wrong then the error
# will appear here instead of in a loop

# modelin
modelin_target_shape = (1, crop_shape[0], crop_shape[1], crop_shape[2], 1)
logger.debug(f"{modelin_target_shape=}")
modelin = crop_data.reshape(modelin_target_shape) 


# In[1]:


# modelout
modelout = model.predict(modelin, batch_size=1)
logger.debug(f"{modelout.shape=}")


# In[ ]:


n_classes = modelout.shape[-1]
logger.debug(f"{n_classes=}")

# probas
crop_probas_target_shape = list(crop_shape) + [n_classes]
logger.debug(f"{crop_probas_target_shape=}")

crop_probas = modelout.reshape(crop_probas_target_shape).astype(probabilities_dtype)
logger.debug(f"{crop_probas.shape=}   {crop_probas.dtype=}")

# preds
crop_preds = crop_probas.argmax(axis=-1).astype(np.int8)
logger.debug(f"{crop_preds.shape=}   {crop_preds.dtype=}")

if debug__save_figs:
    fig, axs = plt.subplots(
        nrows=1, ncols=2,
        figsize=(2 * (sz := 8), sz), 
        dpi=150,
    )

    display = viz.SliceDataPredictionDisplay(
        slice_data=crop_data,
        slice_prediction=crop_preds,
        slice_name=volume.fullname + f".debug.crop-{crop_ijk=}",
        n_classes=n_classes,
    ).plot(axs=axs)

    logger.info(f"Saving figure {(figname := display.title + '.png')=}")
    display.fig_.savefig(
        fname=figs_dir / figname,
        format="png",
        metadata=display.metadata,
    )       
    plt.close()


# # Segment all

# In[ ]:


if debug__materialize_crops:
    
    logger.info("Predicting all crops in advance (materialized version).")
    
    proba_crops_target_shape = list(crops.shape) + [n_classes]
    logger.debug(f"{proba_crops_target_shape=}")

    proba_crops = np.empty(proba_crops_target_shape, dtype=probabilities_dtype)
    logger.debug(f"{proba_crops.shape=} {proba_crops.dtype=}")

    pred_crops = np.empty_like(crops)
    logger.debug(f"{pred_crops.shape=} {pred_crops.dtype=}")

    ijk_iterator = list(get_ijk_iterator())
    n_iterations = len(ijk_iterator)
    logger.debug(f"{n_iterations=}")

    for i, j, k in pbar(ijk_iterator, prefix="crops-segmentation", max_value=n_iterations):
        
        crop_data = crops[i, j, k]
        
        # [model]
        model_in = crop_data.reshape(*modelin_target_shape) 
        model_out = model.predict(model_in)
        proba_crops[i, j, k] = model_out.astype(probabilities_dtype).reshape(crop_probas_target_shape)
        pred_crops[i, j, k] = proba_crops[i, j, k].argmax(axis=-1).astype(np.int8)
    
    if debug__save_processed_crops:
        fname = estimation_volume.debug__crops_probas_path
        logger.info(f"Saving crops probabilities at {fname=}")
        np.save(fname, proba_crops)
        
        fname = estimation_volume.debug__crops_preds_path
        logger.info(f"Saving crops predictions at {fname=}")
        np.save(fname, pred_crops)        


# # Rebuild the volume

# In[ ]:


proba_volume_target_shape = list(volume_shape) + [n_classes]
logger.debug(f"{proba_volume_target_shape=}")

proba_volume = np.zeros(proba_volume_target_shape, dtype=probabilities_dtype)
logger.debug(f"{proba_volume.shape=}")

redundancies_count = np.zeros(proba_volume.shape).max(axis=-1).astype(np.int)  # only one channel
logger.debug(f"{redundancies_count.shape=}")

n_iterations = n_steps[0] * n_steps[1] * n_steps[2]
logger.debug(f"{n_iterations=}")

if debug__materialize_crops:
    # 'F' reshapes with x varying fastest and z slowest
    # this is necessary bcs `crops_coordinates_sequential` is 
    # also reshaped like this
    proba_crops_sequential = proba_crops.reshape(-1, *proba_crops.shape[3:], order='F')  
    logger.debug(f"{proba_crops_sequential.shape=}")

    logger.debug("Summing up the crops' probabilities.")
    for coord, proba_crop in pbar(zip(
        crops_coordinates_sequential,
        proba_crops_sequential,
    ), prefix="sum-probas", max_value=n_iterations):
        slice3d = tuple(slice(*coords_) for coords_ in coord)
        proba_volume[slice3d] += proba_crop
        redundancies_count[slice3d] += np.ones(crop_shape, dtype=np.int)
else:
    logger.debug("Predicting and summing up the crops' probabilities.")
    for coord in pbar(crops_coordinates_sequential, prefix="predict-and-sum-probas", max_value=n_iterations):
        # [model]
        slice3d = tuple(slice(*coords_) for coords_ in coord)
        crop_data = data_volume[slice3d]
        modelin = crop_data.reshape(modelin_target_shape)
        modelout = model.predict(modelin, batch_size=1) 
        proba_volume[slice3d] += modelout.astype(probabilities_dtype).reshape(crop_probas_target_shape)
        redundancies_count[slice3d] += np.ones(crop_shape, dtype=np.int)


# In[ ]:


# check that the min and max probas are coherent with the min/max redundancy
min_proba_sum = proba_volume.min(axis=0).min(axis=0).min(axis=0)
max_proba_sum = proba_volume.max(axis=0).max(axis=0).max(axis=0)
min_redundancy = np.min(redundancies_count)
max_redundancy = np.max(redundancies_count)
assert min_redundancy >= 1, f"{min_redundancy=}"
assert np.all(min_proba_sum >= 0), f"{min_proba_sum=}"
assert np.all(max_proba_sum <= max_redundancy), f"{max_proba_sum=} {max_redundancy=}"

# divide each probability channel by the number of times it was summed (avg proba)
logger.debug(f"Dividing probability redundancies.")
for klass_idx in pbar(range(n_classes), max_value=n_classes, prefix="redundancies-per-class"):
    proba_volume[:, :, :, klass_idx] = proba_volume[:, :, :, klass_idx] / redundancies_count

# check that proba distribs sum to 1
min_proba = proba_volume.min(axis=0).min(axis=0).min(axis=0)
max_proba = proba_volume.max(axis=0).max(axis=0).max(axis=0)
assert np.all(min_proba >= 0), f"{min_proba=}"
assert np.all(max_proba <= 1), f"{max_proba=}"

min_distrib_proba_sum = proba_volume.sum(axis=-1).min()
max_distrib_proba_sum = proba_volume.sum(axis=-1).max()
assert np.isclose(min_distrib_proba_sum, 1, atol=.001), f"{min_distrib_proba_sum=}"
assert np.isclose(max_distrib_proba_sum, 1, atol=.001), f"{max_distrib_proba_sum=}"

pred_volume = proba_volume.argmax(axis=-1).astype("uint8")

logger.debug(f"{pred_volume.shape=}   {pred_volume.min()=}   {pred_volume.max()=}")


# # Save volumes

# In[ ]:


logger.debug(f"Writing probabilities on disk at `{estimation_volume.probabilities_path}`")
np.save(estimation_volume.probabilities_path, proba_volume)


# In[ ]:


logger.debug(f"Writing predictions on disk at `{(str_path := str(estimation_volume.predictions_path))}`")
file_utils.HST_write(pred_volume, str_path)


# In[ ]:


if save_probas_by_class:
    for klass_idx in volume.metadata.labels:
        logger.debug(f"Writing probabilities of class `{klass_idx}` on disk at `{(str_path := str(estimation_volume.get_class_probability_path(klass_idx)))=}`")
        file_utils.HST_write(proba_volume[:, :, :, klass_idx], str_path)


# #### one-z-slice-crops-locations.png
# 
# not kept, search fro `one-z-slice-crops-locations.png` in `process-3d-crops-entire-2d-slice`

# #### debug__materialize_crops
# 
# same for
# `debug__materialize_crops`

# # Save notebook

# In[ ]:


this_nb_name = "process-volume-00.ipynb"
this_dir = os.getcwd()
save_nb_dir = str(estimation_volume.dir)

logger.warning(f"{this_nb_name=}")
logger.warning(f"{this_dir=}")
logger.warning(f"{save_nb_dir=}")

command = f"jupyter nbconvert {this_dir}/{this_nb_name} --output-dir {save_nb_dir} --to html"
os.system(command)

