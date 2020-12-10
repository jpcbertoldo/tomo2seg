http://localhost:8888/lab#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# get_ipython().run_line_magic('load_ext', 'autoreload')


# # In[ ]:


# 'hello'


# # In[ ]:


# get_ipython().run_line_magic('autoreload', '2')

import copy
import dataclasses
import functools
import gc
import itertools
import logging
import operator
import pprint as pprint_module
import time
from functools import partial
from pathlib import Path

import humanize
import numpy as np
import tensorflow as tf
from cnn_segm import keras_custom_loss
from matplotlib import pyplot as plt
from numpy.random import RandomState
from progressbar import progressbar as pbar
from pymicro.file import file_utils
from tensorflow.keras import layers
from tensorflow.keras import optimizers

from tomo2seg import process
from tomo2seg import viz
from tomo2seg.data import EstimationVolume
from tomo2seg.data import Volume
from tomo2seg.logger import get_formatter as logger_get_formatter
from tomo2seg.logger import logger
from tomo2seg.model import Model as Tomo2SegModel

# # Setup

# In[ ]:


logger.setLevel(logging.DEBUG)

random_state = 42
random_state = np.random.RandomState(random_state)


# # Args

# In[ ]:


from tomo2seg.datasets import (
    VOLUME_COMPOSITE_V1 as VOLUME_NAME_VERSION,
#     VOLUME_COMPOSITE_V1_REDUCED as VOLUME_NAME_VERSION,
#     VOLUME_COMPOSITE_NEIGHBOUR as VOLUME_NAME_VERSION,    
#     VOLUME_COMPOSITE_FLEX as VOLUME_NAME_VERSION,    
#     VOLUME_COMPOSITE_BIAXE as VOLUME_NAME_VERSION,    
)

# runid = 1607343593
try:
    runid
except NameError:
    runid = int(time.time())

args = process.ProcessVolumeArgs(
    model_name="unet2d.vanilla03-f16.fold000.1606-505-109",
#     model_name="unet3d.vanilla03-f08.fold000.1606-842-005",
#     model_name="unet2d.vanilla02-f08.fold000.1606-431-664",
#     model_name="unet3d.vanilla03-f16.fold000.1606-750-939",
#     model_name="unet2d.vanilla02-f16.fold000.1606-461-820",
#     model_name="unet2d-sep.vanilla03-f16.fold000.1606-575-226",
#     model_name="unet2d.vanilla03-f16.fold000.1606-505-109",
    model_type=process.ModelType.input2d, 
    
    volume_name=VOLUME_NAME_VERSION[0], 
    volume_version=VOLUME_NAME_VERSION[1], 
    
#     partition_alias=None,
    partition_alias="test",
    
    cropping_strategy=process.CroppingStrategy.maximum_size_reduced_overlap, 
    aggregation_strategy=process.AggregationStrategy.average_probabilities, 
    
    runid=runid,
    probabilities_dtype = np.float16,
    
    opts=process.ProcessVolumeOpts(
        save_probas_by_class = False,
        debug__save_figs = True,
#         override_batch_size = 6,
        save_logs=True,
    ), 
)


# In[ ]:


tomo2seg_model = Tomo2SegModel.build_from_model_name(args.model_name)

volume = Volume.with_check(
    name=args.volume_name, version=args.volume_version
)

partition = volume[args.partition_alias] if args.partition_alias is not None else None

estimation_volume = EstimationVolume.from_objects(
    volume=volume, 
    model=tomo2seg_model, 
    set_partition=partition,
    runid=runid,
)

# this is informal metadata for human use
estimation_volume["aggregation_strategy"] = args.aggregation_strategy.name
estimation_volume["cropping_strategy"] = args.cropping_strategy.name
estimation_volume["probabilities_dtype"] = args.probabilities_dtype.__name__


if args.opts.save_logs:
    fh = logging.FileHandler(estimation_volume.exec_log_path_str)
    fh.setFormatter(logger_get_formatter())
    logger.addHandler(fh)
    logger.info(f"Added a new file handler to the logger. {estimation_volume.exec_log_path_str=}")
    logger.setLevel(logging.DEBUG)

# show inputs

# In[ ]:


logger.info(f"args\n{pprint_module.PrettyPrinter(indent=4, compact=False).pformat(dataclasses.asdict(args))}")
logger.info(f"{estimation_volume=}")
            
logger.debug(f"{volume=}")
logger.debug(f"{partition=}")
logger.debug(f"{tomo2seg_model=}")

if args.model_type == process.ModelType.input2halfd:
    raise NotImplementedError(f"{args.model_type=}")

# # Setup GPUs
# this is here so that the logs will go to the file handler

n_gpus = len(tf.config.list_physical_devices('GPU'))
    
logger.debug(f"{tf.__version__=}")
logger.info(f"Num GPUs Available: {n_gpus}\nThis should be 2 on R790-TOMO.")
logger.debug(f"Should return 2 devices...\n{tf.config.list_physical_devices('GPU')=}")
logger.debug(f"Should return 2 devices...\n{tf.config.list_logical_devices('GPU')=}")

# xla auto-clustering optimization (see: https://www.tensorflow.org/xla#auto-clustering)
# this seems to break the training
tf.config.optimizer.set_jit(False)


# # Load

# ##### gpu distribution strategy

# In[ ]:


# get a distribution strategy to use both gpus (see https://www.tensorflow.org/guide/distributed_training)
# strategy = tf.distribute.MirroredStrategy()  

# there is a bug with MirroredStrategy when you model.predict() with batch_size=1
# https://docs.google.com/document/d/17X1CUvGtlio3pkbKFemSGbF2Qnn0vWAZfCLsgFPoOqg/edit?usp=sharing
one_device = tf.distribute.OneDeviceStrategy(device="/gpu:0" if n_gpus > 0 else "/cpu:0")
# logger.info(f"Because {args.model_type=}, MirroredStrategy cannot be used. Switched to {strategy.__class__.__name__}.")
    
logger.debug(f"{one_device=}")


# ##### model

# In[ ]:


def get_model():
    
    logger.info(f"Loading model from autosaved file: {tomo2seg_model.autosaved_model_path.name}")
    
    model = tf.keras.models.load_model(
        tomo2seg_model.autosaved_model_path_str,
        compile=False
    )
    
    logger.debug("Changing the model's input type to accept any size of crop.")
    
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
    
    return model

with one_device.scope():
    logger.info(f"Loading model with {one_device.__class__.__name__}.")
    model = get_model()


# ##### data

# In[ ]:


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

logger.info(f"Loading data from disk at file: {volume.data_path.name}")

voldata = read_raw(volume.data_path) / 255  # normalize

logger.debug(f"{voldata.shape=}")

if partition is not None:
    logger.debug(f"Cutting data with {partition.alias=}")
    data_volume = partition.get_volume_partition(voldata)

else:
    logger.debug(f"No partition.")
    data_volume = voldata

del voldata

logger.debug(f"{data_volume.shape=}")
logger.debug(f"{data_volume.size=}  ({humanize.intword(data_volume.size)})")


# # Processing

# In[ ]:


if args.opts.debug__save_figs:
    figs_dir = estimation_volume.dir
    
    logger.debug(f"{figs_dir=}")
    figs_dir.mkdir(exist_ok=True)
    
volume_shape = data_volume.shape

logger.info(f"{volume_shape=}")


# ## Shapes

# In[ ]:


MULTIPLE_REQUIREMENT = 16
logger.info(f"{MULTIPLE_REQUIREMENT=}")

MAX_INTERNAL_NVOXELS = max(
    # seen cases
    4 * (8 * 6) * (96**3),
    8 * (16 * 6) * (320**2),  
    3 * (16 * 6) * (800 * 928),
)

MAX_INTERNAL_NVOXELS *= 5/8  # a smaller gpu on other pcs...

logger.info(f"{MAX_INTERNAL_NVOXELS=} ({humanize.intcomma(MAX_INTERNAL_NVOXELS)})")

input_layer = model.layers[0]

logger.debug(f"{input_layer}")

assert (input_layer_class := input_layer.__class__) == tf.keras.layers.InputLayer, f"{input_layer_class=}"

input_nvoxels = functools.reduce(operator.mul, (x for x in input_layer.input_shape[0][1:]))

logger.debug(f"{input_nvoxels=}")


def get_layer_nvoxels(layer) -> int:
    return functools.reduce(operator.mul, (x for x in layer.output_shape[1:]))


internal_nvoxels = [
    get_layer_nvoxels(l)
    for l in model.layers[1:]
]

max_internal_nvoxels = max(internal_nvoxels)

logger.debug(f"{max_internal_nvoxels=} ({humanize.intcomma(max_internal_nvoxels)})")

internal_nvoxel_factor = max_internal_nvoxels / input_nvoxels

logger.debug(f"{internal_nvoxel_factor=}")

assert internal_nvoxel_factor == int(internal_nvoxel_factor), f"{internal_nvoxel_factor=}"

internal_nvoxel_factor = int(internal_nvoxel_factor)

logger.debug(f"{internal_nvoxel_factor=}")

max_batch_nvoxels = int(np.floor(MAX_INTERNAL_NVOXELS / internal_nvoxel_factor))

logger.info(f"{max_batch_nvoxels=} ({humanize.intcomma(max_batch_nvoxels)})")

if args.cropping_strategy == process.CroppingStrategy.maximum_size:
    crop_dims_multiple_16 = process.get_largest_crop_multiple(
        volume_shape, 
        multiple_of=MULTIPLE_REQUIREMENT
    )

elif args.cropping_strategy == process.CroppingStrategy.maximum_size_reduced_overlap:
    # it's not necessarily the real minimum, just an easy way to get a big crop with less overlap
    # get the largest multiple of the requirement above the dimension size / 2
    # that will give a max overlap of 2 * MULTIPLE_REQUIREMENT - 1
    # e.g. with MULTIPLE_REQUIREMENT = 16, the maximum overlap is 31
    crop_dims_multiple_16 = tuple(
        (1 + int((dim / 2) // MULTIPLE_REQUIREMENT)) * MULTIPLE_REQUIREMENT if dim % MULTIPLE_REQUIREMENT != 0 else
        dim
        for dim in volume_shape
    )
    
    logger.info(f"the max overlap in each direction will be {tuple(int(2 * MULTIPLE_REQUIREMENT - s % MULTIPLE_REQUIREMENT) for s in volume_shape)}")
    
else:
    raise ValueError(f"{args.cropping_strategy=}")

logger.debug(f"{crop_dims_multiple_16=} using {args.cropping_strategy=}")

# it has to be multiple of 16 because of the 4 cascaded 2x2-strided 2x2-downsamplings in u-net
if args.model_type == process.ModelType.input2d:
    crop_shape = (
        crop_dims_multiple_16[0],
        crop_dims_multiple_16[1],
        1,
    )

elif args.model_type == process.ModelType.input2halfd:
    raise NotImplemented()
    
elif args.model_type == process.ModelType.input3d:
    crop_shape = crop_dims_multiple_16

logger.debug(f"ideal {crop_shape=} for {args.model_type=} now let's see if the maximum number of voxels is ok...")

crop_shape = process.reduce_dimensions(
    crop_shape,
    max_nvoxels=max_batch_nvoxels,
    multiple_of=MULTIPLE_REQUIREMENT,
)
    
logger.info(f"{crop_shape=}")

crop_nvoxels = functools.reduce(operator.mul, crop_shape)

logger.info(f"{crop_nvoxels=}")

max_batch_size_per_gpu = int(np.floor(max_batch_nvoxels / crop_nvoxels))

logger.info(f"{max_batch_size_per_gpu=}")


# ## Steps and coordinates

# In[ ]:


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
logger.debug(f"{min(x0s)=}, {max(x0s)=}, {len(x0s)=}")
logger.debug(f"{min(y0s)=}, {max(y0s)=}, {len(y0s)=}")
logger.debug(f"{min(z0s)=}, {max(z0s)=}, {len(z0s)=}")


# ### crops coordinates 

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

logger.debug(f"{crops_coordinates.shape=}")

# 'F' reshapes with x varying fastest and z slowest
crops_coordinates_sequential = crops_coordinates.reshape(-1, 3, 2, order='F')  

logger.debug(f"{crops_coordinates_sequential.shape=}")


# ## debug

# ### orthogonal slices plot

# In[ ]:


if args.opts.debug__save_figs:
    
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
    plt.close()


# ### Segment an example

# In[ ]:


crop_ijk = (0, 0, 0)
i, j, k = crop_ijk
crop_coords = crops_coordinates[i, j, k]

logger.info(f"Segmenting one crop for debug {crop_ijk=}")

slice3d = tuple(slice(*coords_) for coords_ in crop_coords)
crop_data = data_volume[slice3d]
del slice3d
    
logger.debug(f"{crop_data.shape=}")

# [model] - i call it with a first crop bc if something goes wrong then the error
# will appear here instead of in a loop

# modelin
modelin_target_shape = (1, crop_shape[0], crop_shape[1], crop_shape[2], 1)

logger.debug(f"{modelin_target_shape=}")

modelin = crop_data.reshape(modelin_target_shape) 

# modelout
modelout = model.predict(
    modelin, 
    batch_size=1,
    steps=1,
    verbose=2,
)

logger.debug(f"{modelout.shape=}")

n_classes = modelout.shape[-1]

assert n_classes == len(volume.metadata.labels), f"{n_classes=} {len(volume.metadata.labels)=}"

# probas
crop_probas_target_shape = list(crop_shape) + [n_classes]

logger.debug(f"{crop_probas_target_shape=}")

crop_probas = modelout.reshape(crop_probas_target_shape).astype(args.probabilities_dtype)

logger.debug(f"{crop_probas.shape=}")
logger.debug(f"{crop_probas.dtype=}")

# preds
crop_preds = crop_probas.argmax(axis=-1).astype(np.int8)

logger.debug(f"{crop_preds.shape=}")
logger.debug(f"{crop_preds.dtype=}")


# In[ ]:


if args.opts.debug__save_figs:
    fig, axs = plt.subplots(
        nrows=3, ncols=2,
        figsize=(2 * (sz := 20), sz), 
        dpi=120,
    )

    display = viz.OrthogonalSlicesPredictionDisplay(
        volume_data=crop_data,
        volume_prediction=crop_preds,
        n_classes=n_classes,
        volume_name=volume.fullname + f".debug.crop-{crop_ijk=}",
    ).plot(axs=axs,)

    logger.info(f"Saving figure {(figname := display.title + '.png')=}")
    display.fig_.savefig(
        fname=figs_dir / figname,
        format="png",
        metadata=display.metadata,
    )       
    plt.close()


# ### Segment a batch with `batch_size=n_gpus` (1 per device)

# In[ ]:


logger.info("Segmenting a batch for debug.")


# In[ ]:


batch_size = max(1, n_gpus)

logger.debug(f"{batch_size=}")


# In[ ]:


mirror = tf.distribute.MirroredStrategy()

with mirror.scope():
    logger.info(f"Loading model with {mirror.__class__.__name__}.")
    model = get_model()


# In[ ]:


batch_coords = crops_coordinates_sequential[:batch_size]

logger.debug(f"{batch_coords.shape=}")

batch_slices = [
    tuple(slice(*coords_) for coords_ in crop_coords)
    for crop_coords in batch_coords
]

logger.debug(f"{batch_slices=}")

batch_data = np.stack([
    data_volume[slice_]
    for slice_ in batch_slices
], axis=0)

logger.debug(f"{batch_data.shape=}")

# [model] - now i call it with a first the mirror strategy to make sure it wont break

# modelin
modelin_target_shape = (batch_size, crop_shape[0], crop_shape[1], crop_shape[2], 1)  # adjust nb. channels

logger.debug(f"{modelin_target_shape=}")

modelin = batch_data.reshape(modelin_target_shape) 

# modelout
modelout = model.predict(
    modelin, 
    batch_size=batch_size,
    steps=1,
    verbose=2,
)

logger.debug(f"{modelout.shape=}")


# In[ ]:


# probas
batch_probas_target_shape = [batch_size] + list(crop_shape) + [n_classes]

logger.debug(f"{batch_probas_target_shape=}")

batch_probas = modelout.reshape(batch_probas_target_shape).astype(args.probabilities_dtype)

logger.debug(f"{batch_probas.shape=}")
logger.debug(f"{batch_probas.dtype=}")

# preds
batch_preds = batch_probas.argmax(axis=-1).astype(np.int8)

logger.debug(f"{batch_preds.shape=}")
logger.debug(f"{batch_preds.dtype=}")


# ### segment batch with `batch_size = n_gpus * max_batch_size_per_gpu`

# In[ ]:


batch_size = max(1, n_gpus) * max_batch_size_per_gpu

logger.debug(f"{batch_size=}")

if args.opts.override_batch_size is not None:
    batch_size = args.opts.override_batch_size
    logger.info(f"{args.opts.override_batch_size=} give ==> replacing the {batch_size=}")

batch_coords = crops_coordinates_sequential[:batch_size]
batch_slices = [
    tuple(slice(*coords_) for coords_ in crop_coords)
    for crop_coords in batch_coords
]
batch_data = np.stack([data_volume[slice_] for slice_ in batch_slices], axis=0)

# [model]
# modelin
modelin_target_shape = (batch_size, crop_shape[0], crop_shape[1], crop_shape[2], 1)  # adjust nb. channels
modelin = batch_data.reshape(modelin_target_shape) 
# modelout
modelout = model.predict(
    modelin, 
    batch_size=batch_size,
    steps=1,
    verbose=2,
)

logger.debug(f"{modelout.shape=}")


# In[ ]:


if args.opts.debug__save_figs:
    
    batch_modelout = modelout
    
    for idx, (crop_data__, crop_probas__) in enumerate(zip(batch_data, batch_modelout)):
        
        fig, axs = plt.subplots(
            nrows=3, ncols=2,
            figsize=(2 * (sz := 20), sz), 
            dpi=120,
        )

        display = viz.OrthogonalSlicesPredictionDisplay(
            volume_data=crop_data__,
            volume_prediction=crop_probas__.argmax(axis=-1).reshape(crop_data__.shape),
            n_classes=n_classes,
            volume_name=volume.fullname + f".debug.batch-segm.{idx=}",
        ).plot(axs=axs,)

        logger.info(f"Saving figure {(figname := display.title + '.png')=}")
        display.fig_.savefig(
            fname=figs_dir / figname,
            format="png",
            metadata=display.metadata,
        )       
        plt.close()


# # Rebuild the volume

# In[ ]:


n_crops = crops_coordinates_sequential.shape[0] 

logger.debug(f"{n_crops=}")

last_batch_size = n_crops % batch_size

if n_gpus > 1:
    assert last_batch_size % n_gpus == 0, f"{last_batch_size=}"

logger.debug(f"{last_batch_size=}")
    
niterations = int(np.floor(crops_coordinates_sequential.shape[0] / batch_size)) 

logger.debug(f"{niterations=}")


# In[ ]:


proba_volume_target_shape = list(volume_shape) + [n_classes]

logger.debug(f"{proba_volume_target_shape=}")

proba_volume = np.zeros(proba_volume_target_shape, dtype=args.probabilities_dtype)

logger.debug(f"{proba_volume.shape=}")

redundancies_count_target_shape = volume_shape

logger.debug(f"{redundancies_count_target_shape=}")

redundancies_count = np.zeros(redundancies_count_target_shape, dtype=np.int8)  # only one channel

logger.debug(f"{redundancies_count.shape=}")

def process_batch(batch_start_, batch_size_):
    batch_end = batch_start_ + batch_size_
    
    batch_coords = crops_coordinates_sequential[batch_start_:batch_end]
    batch_slices = [
        tuple(slice(*coords_) for coords_ in crop_coords)
        for crop_coords in batch_coords
    ]
    batch_data = np.stack([data_volume[slice_] for slice_ in batch_slices], axis=0)
    
    # [model]
    modelin_target_shape = (batch_size_, crop_shape[0], crop_shape[1], crop_shape[2], 1)  # adjust nb. channels
    batch_probas = model.predict(
        batch_data.reshape(modelin_target_shape), 
        batch_size=batch_size_,
        steps=1,
    ).astype(args.probabilities_dtype)

    for slice_, crop_proba in zip(batch_slices, batch_probas):
        proba_volume[slice_] += crop_proba.reshape(crop_probas_target_shape)
        redundancies_count[slice_] += np.ones(crop_shape, dtype=np.int)
        
logger.debug("Predicting and summing up the crops' probabilities.")
for batch_idx in pbar(
    range(niterations), 
    prefix="predict-and-sum-probas", 
    max_value=niterations
):
    batch_start = batch_idx * batch_size
    process_batch(batch_start, batch_size)

if last_batch_size > 0:
    logger.info("Segmenting the last batch")
    batch_start = niterations * batch_size
    process_batch(batch_start, last_batch_size)


# In[ ]:


del data_volume


# In[ ]:


gc.collect()


# ##### sanity checks

# In[ ]:


# check that the min and max probas are coherent with the min/max redundancy
min_proba_sum = proba_volume.min(axis=0).min(axis=0).min(axis=0)
max_proba_sum = proba_volume.max(axis=0).max(axis=0).max(axis=0)
min_redundancy = np.min(redundancies_count)
max_redundancy = np.max(redundancies_count)


# In[ ]:


assert min_redundancy >= 1, f"{min_redundancy=}"
assert np.all(min_proba_sum >= 0), f"{min_proba_sum=}"
assert np.all(max_proba_sum <= max_redundancy), f"{max_proba_sum=} {max_redundancy=}"


# ## Normalize probas

# In[ ]:


# divide each probability channel by the number of times it was summed (avg proba)
logger.debug(f"Dividing probability redundancies.")
for klass_idx in pbar(range(n_classes), max_value=n_classes, prefix="redundancies-per-class"):
    proba_volume[:, :, :, klass_idx] = proba_volume[:, :, :, klass_idx] / redundancies_count


# In[ ]:


del redundancies_count


# In[ ]:


gc.collect()


# In[ ]:


# this makes it more stable so that the sum is 1
proba_volume[:, :, :] /= proba_volume[:, :, :].sum(axis=-1, keepdims=True) 


# ##### sanity checks

# In[ ]:


# check that proba distribs sum to 1
min_proba = proba_volume.min(axis=0).min(axis=0).min(axis=0)
max_proba = proba_volume.max(axis=0).max(axis=0).max(axis=0)


# In[ ]:


assert np.all(min_proba >= 0), f"{min_proba=}"
assert np.all(max_proba <= 1), f"{max_proba=}"


# In[ ]:


min_distrib_proba_sum = proba_volume.sum(axis=-1).min()
max_distrib_proba_sum = proba_volume.sum(axis=-1).max()


# In[ ]:


assert np.isclose(min_distrib_proba_sum, 1, atol=.001), f"{min_distrib_proba_sum=}"
assert np.isclose(max_distrib_proba_sum, 1, atol=.001), f"{max_distrib_proba_sum=}"


# In[ ]:


gc.collect()


# # proba 2 pred

# In[ ]:


pred_volume = np.empty(proba_volume.shape[:3], dtype="uint8")


# In[ ]:


np.argmax(proba_volume, axis=-1, out=pred_volume)

logger.debug(f"{pred_volume.shape=}")
logger.debug(f"{pred_volume.min()=}")
logger.debug(f"{pred_volume.max()=}")


# # Save

# In[ ]:


logger.debug(f"Writing probabilities on disk at `{estimation_volume.probabilities_path}`")
np.save(estimation_volume.probabilities_path, proba_volume)


# In[ ]:


if args.opts.save_probas_by_class:
    for klass_idx in volume.metadata.labels:
        logger.debug(f"Writing probabilities of class `{klass_idx}` on disk at `{(str_path := str(estimation_volume.get_class_probability_path(klass_idx)))=}`")
        file_utils.HST_write(proba_volume[:, :, :, klass_idx], str_path)


# In[ ]:


logger.debug(f"Writing predictions on disk at `{(str_path := str(estimation_volume.predictions_path))}`")
file_utils.HST_write(pred_volume, str_path)


# #### one-z-slice-crops-locations.png
# 
# not kept, search fro `one-z-slice-crops-locations.png` in `process-3d-crops-entire-2d-slice`

# #### debug__materialize_crops
# 
# same for
# `debug__materialize_crops`

# # Save notebook

# In[ ]:


# this_nb_name = "process-volume-02.ipynb"
# this_dir = os.getcwd()
# save_nb_dir = str(estimation_volume.dir)

# logger.warning(f"{this_nb_name=}")
# logger.warning(f"{this_dir=}")
# logger.warning(f"{save_nb_dir=}")

# command = f"jupyter nbconvert {this_dir}/{this_nb_name} --output-dir {save_nb_dir} --to html"
# os.system(command)


# In[ ]:





# In[ ]:




