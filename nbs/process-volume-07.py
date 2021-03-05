# coding: utf-8

import os
import copy
import dataclasses
from dataclasses import asdict
import functools
import gc
import itertools
import logging
import operator
import pprint as pprint_module
import time
from functools import partial
from pathlib import Path
import sys

import humanize
import numpy as np
import tensorflow as tf
from cnn_segm import keras_custom_loss
from matplotlib import pyplot as plt
from numpy.random import RandomState
from progressbar import progressbar as pbar
from pymicro.file import file_utils
import socket
from tensorflow.keras import layers
from tensorflow.keras import optimizers

from tomo2seg.process import reduce_dimensions 
from tomo2seg.args import ProcessVolumeArgs as Args
from tomo2seg import viz
from tomo2seg.data import EstimationVolume
from tomo2seg.data import Volume
from tomo2seg.logger import add_file_handler as logger_add_file_handler
from tomo2seg.logger import dict2str
from tomo2seg.logger import logger
from tomo2seg.model import Model as Tomo2SegModel
from tomo2seg import utils as tomo2seg_utils
from tomo2seg import slackme
from tomo2seg import slack
from tomo2seg import volume_sequence
from tomo2seg import hosts as t2s_hosts
from tomo2seg import datasets as t2s_datasets 


# model_name = sys.argv[1]
# [manual-input]
# model_name = "paper-unet-2d.full-f16.fold000.1611-664-168"
import sys 
model_name = sys.argv[1]

try:
    
    # [manual-input]
    volume_name_version = t2s_datasets.VOLUME_COMPOSITE_V1

    args = Args.setup00_process_test(
        # 3d
        model_type = Args.ModelType.input2d, 
        model_name = model_name,

        volume_name=volume_name_version[0], 
        volume_version=volume_name_version[1], 

        script_name = "process-volume-07.ipynb",
        host = None,  # get from socket.hostname

        runid = None,  # default is time.time()
        random_state_seed = 42,  # None = auto value
    )

    logger.info(f"args={dict2str(asdict(args))}")

    # build `tomo2seg` objects 

    tomo2seg_model = Tomo2SegModel.build_from_model_name(args.model_name)

    volume = Volume.with_check(
        name=args.volume_name, 
        version=args.volume_version
    )

    partition = volume[args.partition_alias] if args.partition_alias is not None else None

    estimation_volume = EstimationVolume.from_objects(
        volume=volume, 
        model=tomo2seg_model, 
        set_partition=partition,
        runid=args.runid,
    )

    if args.opts.save_logs:
        logger_add_file_handler(logger, estimation_volume.exec_log_path)

    # this is informal metadata for human use
    estimation_volume["process_volume_args"] = dataclasses.asdict(args)


    # show args

    logger.info("showing args")

    logger.info(f"{estimation_volume=}")
    logger.info(f"{estimation_volume.fullname=}")
    logger.info(f"{estimation_volume.dir=}")

    logger.debug(f"{volume=}")
    logger.debug(f"{partition=}")
    logger.debug(f"{tomo2seg_model=}")
    logger.debug(f"{tomo2seg_model.name=}")

    # # Setup

    logger.info("set up")

    logger.setLevel(logging.DEBUG)
    random_state = np.random.RandomState(args.random_state_seed)

    n_gpus = len(tf.config.list_physical_devices('GPU'))
    estimation_volume["n_gpus"] = n_gpus

    tf_version = tf.__version__
    logger.info(f"{tf_version=}")
    estimation_volume["tf_version"] = tf_version

    logger.info(f"Num GPUs Available: {n_gpus}\nThis should be:\n\t" + '\n\t'.join(['2 on R790-TOMO', '1 on akela', '1 on hathi', '1 on krilin']))

    logger.debug(
        "physical GPU devices:\n\t" + "\n\t".join(map(str, tf.config.list_physical_devices('GPU'))) + "\n" +
        "logical GPU devices:\n\t" + "\n\t".join(map(str, tf.config.list_logical_devices('GPU'))) 
    )

    # xla auto-clustering optimization (see: https://www.tensorflow.org/xla#auto-clustering)
    # this seems to break the training
    tf.config.optimizer.set_jit(False)

    logger.info(f"{dict2str(asdict(args.host))}")

    MAX_INTERNAL_NVOXELS = int(
        args.host.gpu_max_memory_factor * t2s_hosts.MAX_INTERNAL_NVOXELS
    )

    logger.info(f"{MAX_INTERNAL_NVOXELS=} ({humanize.intcomma(MAX_INTERNAL_NVOXELS)})")
    estimation_volume["MAX_INTERNAL_NVOXELS"] = MAX_INTERNAL_NVOXELS


    if args.opts.debug__save_figs:
        figs_dir = estimation_volume.dir / "debug_figs"
        logger.debug(f"{figs_dir=}")
        figs_dir.mkdir(exist_ok=True)


    # # Load

    logger.info("load - start")

    # ##### `tf.distribute.OneDeviceStrategy` 
    # 
    # first just open the model to see that everything goes right

    # get a distribution strategy to use both gpus (see https://www.tensorflow.org/guide/distributed_training)
    one_device = tf.distribute.OneDeviceStrategy(device="/gpu:0" if n_gpus > 0 else "/cpu:0")
    logger.debug(f"{one_device=}")


    # ##### model

    def get_model():

        try:
            best_autosaved_model_path = tomo2seg_model.autosaved2_best_model_path  # it's a property
            assert best_autosaved_model_path is not None, "no-autosaved2"
        except ValueError as ex:

            if ex.args[0] != "min() arg is an empty sequence":
                raise ex

            logger.warning(f"{tomo2seg_model.name=} did not use autosaved2 apparently, falling back to autosaved.")
            best_autosaved_model_path = tomo2seg_model.autosaved_model_path

        except AssertionError as ex:

            if ex.args[0] != "no-autosaved2":
                raise ex

            logger.warning(f"{tomo2seg_model.name=} did not use autosaved2 apparently, falling back to autosaved.")
            best_autosaved_model_path = tomo2seg_model.autosaved_model_path

        print(best_autosaved_model_path)
        logger.info(f"Loading model from autosaved file: {best_autosaved_model_path.name}")

        model = tf.keras.models.load_model(
            str(best_autosaved_model_path),
            compile=False
        )

        logger.debug("Changing the model's input type to accept any size of crop.")

        in_ = model.layers[0]
        in_shape = in_.input_shape[0]
        input_n_channels = in_shape[-1]

        logger.debug(f"{input_n_channels=}")

        if input_n_channels > 1:

            if args.model_type == Args.ModelType.input2halfd:
                if len(in_shape) != 4:
                    raise f"len({in_shape=}) > 4, so this model must be multi-channel. Not supported yet..."
            else:
                raise NotImplementedError(f"{input_n_channels=} > 1")

        # make it capable of getting any dimension in the input
        # "-2" = 1 for the batch size, 1 for the nb.channels
        anysize_target_shape = (len(in_shape) - 2) * [None] + [input_n_channels] 
        logger.debug(f"{anysize_target_shape=}")

        anysize_input = layers.Input(
            shape=anysize_target_shape,
            name="input_any_image_size"
        )
        logger.debug(f"{anysize_input=}")

        model.layers[0] = anysize_input

        # this doesn't really matter bc this script will not fit the model
        optimizer = optimizers.Adam()
        loss_func = keras_custom_loss.jaccard2_loss

        logger.debug("Starting model compilation")
        model.compile(loss=loss_func, optimizer=optimizer)
        logger.debug("Done!")

        return model

    with one_device.scope():
        logger.info(f"Loading model with {one_device.__class__.__name__}.")
        model = get_model()
        logger.info("done")


    # ##### data

    logger.info(f"Loading data from disk at file: {volume.data_path.name}")
    logger.debug(f"{volume.data_path=}")

    normalization_factor = volume_sequence.NORMALIZE_FACTORS[volume.metadata.dtype]

    logger.debug(f"{normalization_factor=}")

    data_volume = file_utils.HST_read(
        str(volume.data_path),  # it doesn't accept paths...
        autoparse_filename=False,  # the file names are not properly formatted
        data_type=volume.metadata.dtype,
        dims=volume.metadata.dimensions,
        verbose=True,
    ) / normalization_factor  # normalize

    logger.debug(f"{data_volume.shape=}")

    if partition is not None:

        logger.info(f"Cutting data with {partition.alias=}")
        logger.debug(f"{partition=}")

        data_volume = partition.get_volume_partition(data_volume)

    else:
        logger.debug(f"No partition. The whole volume will be processed.")

    logger.info("done")


    # modify the data if necessary 
    # 
    # mostly the 2halfd...

    if args.model_type == Args.ModelType.input2halfd:

        try:
            # this is to prevent running the padding twice in the notebook
            half_pad

        except NameError:

                logger.warning("Modifying the data to add a 'reflect' half padding to the data. Only z-layers 2.5d models are supported!")

                nlayers_2halfd = model.layers[0].input_shape[0][-1]

                predicted_layer_idx_2halfd = nlayers_2halfd // 2

                slice_2halfd_data_predicted_layer = slice(predicted_layer_idx_2halfd, predicted_layer_idx_2halfd + 1)

                logger.debug(f"{nlayers_2halfd=}")
                logger.debug(f"{predicted_layer_idx_2halfd=}")
                logger.debug(f"{slice_2halfd_data_predicted_layer=}")

                assert nlayers_2halfd % 2 == 1, f"{nlayers_2halfd=} should be an odd number"

                half_pad = (nlayers_2halfd - 1) // 2

                logger.debug(f"{half_pad=}")

                data_volume = np.pad(
                    data_volume, 
                    pad_width=((0, 0), (0, 0), (half_pad, half_pad)),
                    mode="reflect",
                )

                logger.debug(f"{data_volume.shape=}")    
                estimation_volume["volume_is_padded"] = True
                estimation_volume["half_pad"] = half_pad

        else:
            logger.debug("Padding already applied.")


    volume_shape = data_volume.shape
    logger.info(f"{volume_shape=}")
    logger.info(f"{data_volume.size=}  ({humanize.intword(data_volume.size)})")
    estimation_volume["volume_shape"] = volume_shape

    logger.info("load - end")


    # # Processing

    # ## Shapes

    # how many voxels the gpus can take in a single batch?

    logger.info("processing - shapes - start")

    logger.info(f"{args.model_shape_min_multiple_requirement=}")
    logger.info(f"{MAX_INTERNAL_NVOXELS=} ({humanize.intcomma(MAX_INTERNAL_NVOXELS)})")

    internal_nvoxel_factor = tomo2seg_utils.get_model_internal_nvoxel_factor(model)

    logger.debug(f"{internal_nvoxel_factor=}")

    max_batch_nvoxels = int(np.floor(MAX_INTERNAL_NVOXELS / internal_nvoxel_factor))

    logger.info(f"{max_batch_nvoxels=} ({humanize.intcomma(max_batch_nvoxels)})")


    # figure out the crop shape

    logger.info(f"Using args.cropping_strategy={args.cropping_strategy.name} to find a suitable crop size.")

    if args.cropping_strategy == Args.CroppingStrategy.maximum_size:

        crop_dims_multiple = process.get_largest_crop_multiple(
            volume_shape, 
            multiple_of=args.model_shape_min_multiple_requirement
        )

    elif args.cropping_strategy == Args.CroppingStrategy.maximum_size_reduced_overlap:

        # it's not necessarily the real minimum, just an easy way to get a big crop with less overlap
        # get the largest multiple of the requirement above the dimension size / 2
        # that will give a max overlap of 2 * MULTIPLE_REQUIREMENT - 1
        # e.g. with MULTIPLE_REQUIREMENT = 16, the maximum overlap is 31
        _mult = args.model_shape_min_multiple_requirement
        crop_dims_multiple = tuple(
            (1 + int((dim / 2) // _mult)) * _mult if dim % _mult != 0 else
            dim
            for dim in volume_shape
        )

        def max_overlap(size):
            overlap = int(2 * _mult - size % _mult)
            return overlap if overlap < 32 else 0 

        logger.info(f"the max overlap in each direction will be {tuple(max_overlap(s) for s in volume_shape)}")

    else:
        raise ValueError(f"{args.cropping_strategy=}")

    logger.debug(f"{crop_dims_multiple=}")


    # adjust the crop dimension if necessary

    # it has to be multiple of 16 because of the 4 cascaded 2x2-strided 2x2-downsamplings in u-net
    if args.model_type == Args.ModelType.input2d:
        crop_shape = (
            crop_dims_multiple[0],
            crop_dims_multiple[1],
            1,
        )

    elif args.model_type == Args.ModelType.input2halfd:
        crop_shape = (
            crop_dims_multiple[0],
            crop_dims_multiple[1],
            nlayers_2halfd,
        )

    elif args.model_type == Args.ModelType.input3d:
        crop_shape = crop_dims_multiple

    else:
        raise ValueError(f"{args.model_type=}")

    logger.debug(f"ideal {crop_shape=} for {args.model_type=} now let's see if the maximum number of voxels is ok...")

    crop_shape = reduce_dimensions(
        crop_shape,
        max_nvoxels=max_batch_nvoxels,
        multiple_of=args.model_shape_min_multiple_requirement,
    )

    logger.info(f"{crop_shape=} ")

    crop_nvoxels = functools.reduce(operator.mul, crop_shape)

    logger.info(f"{crop_nvoxels=} ({humanize.intcomma(crop_nvoxels)})")

    max_batch_size_per_gpu = int(np.floor(max_batch_nvoxels / crop_nvoxels))

    logger.info(f"{max_batch_size_per_gpu=}")

    estimation_volume["crop_shape"] = crop_shape
    estimation_volume["crop_nvoxels"] = crop_nvoxels

    logger.info("processing - shapes - end")


    # ## Steps and coordinates

    logger.info("processing - steps and coordinates - start")

    def get_coordinates_in(axis_: int):
        assert 0 <= axis_ <= 2, f"{axis_=}"

        vol_dim = volume_shape[axis_]
        crop_dim = crop_shape[axis_]

        start = 0

        if axis_ == 2 and args.model_type == Args.ModelType.input2halfd:
            n = vol_dim - 2 * half_pad
            end = n - 1

        elif axis_ == 2 and args.model_type == Args.ModelType.input2d:
            assert crop_dim == 1, f"{crop_dim=}"
            end = vol_dim - crop_dim  # vol_dim - 1
            n = vol_dim  # vol_dim / 1 = vol_dim

        else:
            end = vol_dim - crop_dim
            n = int(np.ceil(vol_dim / crop_dim))

        return tuple(map(int, np.linspace(start, end, n)))

    # coordinates (xs, ys, and zs) of the front upper left corners of the crops
    x0s, y0s, z0s = tuple(
        get_coordinates_in(axis_=axxis_)
        for axxis_ in range(3)
    )

    logger.debug(x0s_ := f"{min(x0s)=}, {max(x0s)=}, {len(x0s)=}")
    logger.debug(y0s_ := f"{min(y0s)=}, {max(y0s)=}, {len(y0s)=}")
    logger.debug(z0s_ := f"{min(z0s)=}, {max(z0s)=}, {len(z0s)=}")

    ncrops = len(x0s) * len(y0s) * len(z0s)
    logger.debug(f"{ncrops=}")

    estimation_volume["x0s"] = x0s_
    estimation_volume["y0s"] = y0s_
    estimation_volume["z0s"] = z0s_
    estimation_volume["ncrops"] = ncrops

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

    logger.info("processing - steps and coordinates - end")


    # ## debug

    logger.info("debg - start")


    # ### orthogonal slices plot

    if args.opts.debug__save_figs:

        logger.info("plotting debug orthogonal slice")

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

        logger.info("plotting debug orthogonal slice DONE")



    # ### Segment an example

    logger.info("segmenting an example - start")

    crop_ijk = (0, 0, 0)
    i, j, k = crop_ijk
    crop_coords = crops_coordinates[i, j, k]

    logger.info(f"Segmenting one crop for debug {crop_ijk=}")

    crop_data = data_volume[tuple(slice(*coords_) for coords_ in crop_coords)]

    logger.debug(f"{crop_data.shape=}")

    # [model] - i call it with a first crop bc if something goes wrong then the error
    # will appear here instead of in a loop

    modelin_target_shape = (1, crop_shape[0], crop_shape[1], crop_shape[2], 1)
    logger.debug(f"{modelin_target_shape=}")

    # modelin
    modelin = crop_data.reshape(modelin_target_shape) 

    # modelout
    logger.debug("mode.predict")
    modelout = model.predict(
        modelin, 
        batch_size=1,
        steps=1,
        verbose=2,
    )
    logger.debug("mode.predict done")

    logger.debug(f"{modelout.shape=}")

    n_classes = modelout.shape[-1]

    assert n_classes == len(volume.metadata.labels), f"{n_classes=} {len(volume.metadata.labels)=}"

    if args.model_type == Args.ModelType.input2halfd:
        crop_probas_target_shape = list(crop_shape[:2]) + [1] + [n_classes]

    else:
        crop_probas_target_shape = list(crop_shape) + [n_classes]

    logger.debug(f"{crop_probas_target_shape=}")

    # probas
    crop_probas = modelout.reshape(crop_probas_target_shape).astype(args.probabilities_dtype)

    logger.debug(f"{crop_probas.shape=}")
    logger.debug(f"{crop_probas.dtype=}")

    # preds
    crop_preds = crop_probas.argmax(axis=-1).astype(np.int8)

    logger.debug(f"{crop_preds.shape=}")
    logger.debug(f"{crop_preds.dtype=}")


    if args.opts.debug__save_figs:

        logger.info("plotting debug predicted slice")

        fig, axs = plt.subplots(
            nrows=3, ncols=2,
            figsize=(2 * (sz := 20), sz), 
            dpi=120,
        )

        viz_crop_data = (
            crop_data[:, :, slice_2halfd_data_predicted_layer] 
            if args.model_type == Args.ModelType.input2halfd else 
            crop_data
        )

        display = viz.OrthogonalSlicesPredictionDisplay(
            volume_data=viz_crop_data,
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

        logger.info("plotting debug predicted slice DONE")

    logger.info("segmenting an example - end")


    # ### Segment a batch with `batch_size=n_gpus` (1 per device)

    logger.info("Segmenting a batch with a single instance per gpu for debug.")

    batch_size = max(1, n_gpus)
    logger.debug(f"{batch_size=}")

    mirror = tf.distribute.MirroredStrategy()

    with mirror.scope():
        logger.info(f"Loading model with {mirror.__class__.__name__}.")
        model = get_model()

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

    batch_modelin_target_shape = tuple([batch_size] + list(modelin_target_shape[1:]))  # adjust nb. channels
    batch_probas_target_shape = tuple([batch_size] + list(crop_probas_target_shape))

    logger.debug(f"{batch_modelin_target_shape=}")
    logger.debug(f"{batch_probas_target_shape=}")

    # modelin
    modelin = batch_data.reshape(batch_modelin_target_shape) 

    # modelout
    logger.debug("model.predict")
    modelout = model.predict(
        modelin, 
        batch_size=batch_size,
        steps=1,
        verbose=2,
    )
    logger.debug("model.predict done")

    logger.debug(f"{modelout.shape=}")

    # probas
    batch_probas = modelout.reshape(batch_probas_target_shape).astype(args.probabilities_dtype)

    logger.debug(f"{batch_probas.shape=}")
    logger.debug(f"{batch_probas.dtype=}")

    # preds
    batch_preds = batch_probas.argmax(axis=-1).astype(np.int8)

    logger.debug(f"{batch_preds.shape=}")
    logger.debug(f"{batch_preds.dtype=}")

    logger.info("Segmenting a batch with a single instance per gpu for debug - DONE")


    # ### segment batch with `batch_size = n_gpus * max_batch_size_per_gpu`

    logger.info("Segmenting a batch with as many instances per gpu as possible for debug.")

    batch_size = max(1, n_gpus) * max_batch_size_per_gpu
    logger.debug(f"{batch_size=}")

    if args.opts.override_batch_size is not None:
        batch_size = args.opts.override_batch_size
        logger.info(f"{args.opts.override_batch_size=} give ==> replacing the {batch_size=}")

    # usefull for debug 
    batch_start = 0
    batch_end = batch_start + batch_size

    logger.debug(f"{batch_start=}")
    logger.debug(f"{batch_end=}")

    batch_coords = crops_coordinates_sequential[batch_start:batch_end]
    batch_slices = [
        tuple(slice(*coords_) for coords_ in crop_coords)
        for crop_coords in batch_coords
    ]
    batch_data = np.stack([data_volume[slice_] for slice_ in batch_slices], axis=0)

    logger.debug(f"{batch_data.shape=}")

    batch_modelin_target_shape = tuple([batch_size] + list(modelin_target_shape[1:]))  # adjust nb. channels
    batch_probas_target_shape = tuple([batch_size] + list(crop_probas_target_shape))

    logger.debug(f"{batch_modelin_target_shape=}")
    logger.debug(f"{batch_probas_target_shape=}")

    # [model]
    # modelin
    modelin = batch_data.reshape(batch_modelin_target_shape)

    # modelout
    logger.debug("model.predict")
    modelout = model.predict(
        modelin, 
        batch_size=batch_size,
        steps=1,
        verbose=2,
    )
    logger.debug("model.predict done")

    logger.debug(f"{modelout.shape=}")

    # probas
    batch_probas = modelout.reshape(batch_probas_target_shape).astype(args.probabilities_dtype)

    logger.debug(f"{batch_probas.shape=}")

    # preds
    batch_preds = batch_probas.argmax(axis=-1).astype(np.int8)

    logger.debug(f"{batch_preds.shape=}")

    logger.info("Segmenting a batch with as many instances per gpu as possible for debug - DONE.")

    MAX_DEBUG_FIGS = 10

    if args.opts.debug__save_figs:

        logger.info("plotting debug batch of predicted slices")

        batch_modelout = modelout

        indices = (
            range(batch_size) if batch_size <= MAX_DEBUG_FIGS else 
            map(int, np.linspace(0, batch_size, MAX_DEBUG_FIGS,)[:-1])
        )

        for idx in indices:
            crop_data__ = batch_data[idx]
            crop_preds__ = batch_preds[idx]

            fig, axs = plt.subplots(
                nrows=3, ncols=2,
                figsize=(2 * (sz := 20), sz), 
                dpi=120,
            )

            viz_crop_data = (
                crop_data__[:, :, slice_2halfd_data_predicted_layer] 
                if args.model_type == Args.ModelType.input2halfd else 
                crop_data__
            )

            display = viz.OrthogonalSlicesPredictionDisplay(
                volume_data=viz_crop_data,
                volume_prediction=crop_preds__,
                n_classes=n_classes,
                volume_name=volume.fullname + f".debug.batch-segm.{idx=:04d}",
            ).plot(axs=axs,)

            logger.info(f"Saving figure {(figname := display.title + '.png')=}")
            display.fig_.savefig(
                fname=figs_dir / figname,
                format="png",
                metadata=display.metadata,
            )       
            plt.close()

        logger.info("plotting debug batch of predicted slices - DONE")



    # # Rebuild the volume

    logger.info("rebuilding volume - start")

    last_batch_size = ncrops % batch_size

    logger.debug(f"{last_batch_size=}")

    niterations = int(np.floor(crops_coordinates_sequential.shape[0] / batch_size)) 

    logger.debug(f"{niterations=}")

    if args.model_type == Args.ModelType.input2halfd:
        proba_volume_target_shape = list(volume_shape[:2]) + [volume_shape[2] - 2 * half_pad] + [n_classes]

    else:
        proba_volume_target_shape = list(volume_shape) + [n_classes]

    redundancies_count_target_shape = proba_volume_target_shape[:3]

    logger.debug(f"{proba_volume_target_shape=}")
    logger.debug(f"{redundancies_count_target_shape=}")

    proba_volume = np.zeros(proba_volume_target_shape, dtype=args.probabilities_dtype)
    logger.debug(f"{proba_volume.shape=}")

    redundancies_count = np.zeros(redundancies_count_target_shape, dtype=np.int8)  # only one channel
    logger.debug(f"{redundancies_count.shape=}")

    logger.debug(f"{niterations=}")

    def process_batch(batch_start_, batch_end_):

        global model

        batch_size_ = batch_end_ - batch_start_

        # adjust nb. channels
        batch_modelin_target_shape_ = tuple([batch_size_] + list(modelin_target_shape[1:]))  
        batch_probas_target_shape_ = tuple([batch_size_] + list(crop_probas_target_shape))

        batch_coords = crops_coordinates_sequential[batch_start_:batch_end_]
        batch_slices = [
            tuple(slice(*coords_) for coords_ in crop_coords)
            for crop_coords in batch_coords
        ]
        batch_data = np.stack([data_volume[slice_] for slice_ in batch_slices], axis=0)

        # [model] 
        batch_probas = model.predict(
            batch_data.reshape(batch_modelin_target_shape_), 
            batch_size=batch_size_,
            steps=1,
        ).astype(args.probabilities_dtype).reshape(batch_probas_target_shape_)

        for slice_, crop_proba in zip(batch_slices, batch_probas):

            if args.model_type == Args.ModelType.input2halfd:
                # keep x and y as is, but reduce z
                slice_ = tuple(
                    list(slice_[:2]) +
                    [slice(slice_[2].start, slice_[2].start + 1)]
                )

            proba_volume[slice_] += crop_proba
            redundancies_count[slice_] += np.ones(crop_proba.shape[:-1], dtype=np.int)

    logger.info("Predicting and summing up the crops' probabilities.")
    for batch_idx in pbar(
        range(niterations), 
        prefix="predict-and-sum-probas", 
        max_value=niterations
    ):
        batch_start = batch_idx * batch_size
        batch_end = batch_start + batch_size

        try:
            process_batch(batch_start, batch_end)

        except Exception as ex:
            logger.debug(f"{batch_idx=} {batch_start=} {batch_end=}")
            logger.exception(ex)
            raise ex

    if last_batch_size > 0:
        logger.info("Segmenting the last batch")

        try:
            if n_gpus > 1:
                assert last_batch_size % n_gpus == 0, f"{ncrops=} {batch_size=} {last_batch_size=}"

        except AssertionError as ex:

            logger.warning(
                "The size of the last batch is not a multiple of the number of GPUs, "
                "so it will be processed with single device strategy."
            )

            with one_device.scope():
                logger.info(f"Loading model with {one_device.__class__.__name__}.")
                model = get_model()

        batch_start = niterations * batch_size
        batch_end = batch_start + last_batch_size

        process_batch(batch_start, batch_end)

    logger.info("Predicting and summing up the crops' probabilities - DONE.")


    # ## sanity checks

    logger.info("proba sanity checks")

    # check that the min and max probas are coherent with the min/max redundancy
    min_proba_sum = proba_volume.min(axis=0).min(axis=0).min(axis=0)
    max_proba_sum = proba_volume.max(axis=0).max(axis=0).max(axis=0)
    min_redundancy = np.min(redundancies_count)
    max_redundancy = np.max(redundancies_count)





    assert min_redundancy >= 1, f"{min_redundancy=}"
    assert np.all(min_proba_sum >= 0), f"{min_proba_sum=}"
    assert np.all(max_proba_sum <= max_redundancy), f"{max_proba_sum=} {max_redundancy=}"

    logger.info("proba sanity checks done")


    # ## Normalize probas

    # divide each probability channel by the number of times it was summed (avg proba)
    logger.debug(f"Dividing probability redundancies.")

    for klass_idx in pbar(range(n_classes), max_value=n_classes, prefix="redundancies-per-class"):
        proba_volume[:, :, :, klass_idx] = proba_volume[:, :, :, klass_idx] / redundancies_count

    # this makes it more stable so that the sum is 1
    proba_volume[:, :, :] /= proba_volume[:, :, :].sum(axis=-1, keepdims=True) 

    logger.debug(f"Dividing probability redundancies done.")


    # ## sanity checks

    logger.info("proba sanity checks 2")

    # check that proba distribs sum to 1
    min_proba = proba_volume.min(axis=0).min(axis=0).min(axis=0)
    max_proba = proba_volume.max(axis=0).max(axis=0).max(axis=0)

    assert np.all(min_proba >= 0), f"{min_proba=}"
    assert np.all(max_proba <= 1), f"{max_proba=}"

    min_distrib_proba_sum = proba_volume.sum(axis=-1).min()
    max_distrib_proba_sum = proba_volume.sum(axis=-1).max()

    assert np.isclose(min_distrib_proba_sum, 1, atol=.001), f"{min_distrib_proba_sum=}"
    assert np.isclose(max_distrib_proba_sum, 1, atol=.001), f"{max_distrib_proba_sum=}"

    logger.info("proba sanity checks 2 done")


    # ## proba 2 pred

    logger.info("proba 2 pred - start")

    pred_volume = np.empty(proba_volume.shape[:-1], dtype="uint8")

    np.argmax(proba_volume, axis=-1, out=pred_volume)

    logger.debug(f"{pred_volume.shape=}")
    logger.debug(f"{pred_volume.min()=}")
    logger.debug(f"{pred_volume.max()=}")

    if args.opts.debug__save_figs:

        fig, axs = plt.subplots(
            nrows=3, ncols=2,
            figsize=(2 * (sz := 20), sz), 
            dpi=120,
        )

        viz_data = (
            data_volume[:, :, half_pad:-half_pad] 
            if args.model_type == Args.ModelType.input2halfd else 
            data_volume
        )

        display = viz.OrthogonalSlicesPredictionDisplay(
            volume_data=viz_data,
            volume_prediction=pred_volume,
            n_classes=n_classes,
            volume_name=volume.fullname + f".debug.predicted-volume.{idx=}",
        ).plot(axs=axs,)

        logger.info(f"Saving figure {(figname := display.title + '.png')=}")
        display.fig_.savefig(
            fname=figs_dir / figname,
            format="png",
            metadata=display.metadata,
        )       
        plt.close()

    logger.info("proba 2 pred - end")
    logger.info("rebuilding volume - end")


    # # Save

    logger.debug(f"Writing probabilities on disk at `{estimation_volume.probabilities_path}`")
    np.save(estimation_volume.probabilities_path, proba_volume)

    if args.opts.save_probas_by_class:
        for klass_idx in volume.metadata.labels:
            logger.debug(f"Writing probabilities of class `{klass_idx}` on disk at `{(str_path := str(estimation_volume.get_class_probability_path(klass_idx)))=}`")
            file_utils.HST_write(proba_volume[:, :, :, klass_idx], str_path)

    pred_volume.size * pred_volume.itemsize / 1024 ** 2

    logger.debug(f"Writing predictions on disk at `{(str_path := str(estimation_volume.predictions_path))}`")
    file_utils.HST_write(pred_volume, str_path)

    # # Slack (:

    slack.notify("process-volume finished!")

except Exception as ex:
    
    slack.notify_exception(ex, args.host.hostname)
