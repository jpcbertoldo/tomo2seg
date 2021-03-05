# coding: utf-8

# # Imports
# 

import os
from dataclasses import asdict
import logging
from typing import *
import socket

import humanize
from matplotlib import pyplot as plt
import numpy as np
from numpy.random import RandomState
import pandas as pd
from pymicro.file import file_utils
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import utils
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks as keras_callbacks

from adabelief_tf import AdaBeliefOptimizer

from tomo2seg import datasets as t2s_datasets
from tomo2seg import slack
from tomo2seg import modular_unet
from tomo2seg.logger import logger, dict2str, add_file_handler
from tomo2seg import viz
from tomo2seg.data import Volume
from tomo2seg.volume_sequence import (
    MetaCrop3DGenerator, VolumeCropSequence,
    SequentialGridPosition
)
from tomo2seg import volume_sequence as t2s_volseq
from tomo2seg.model import Model as T2SModel, Type as T2SModelType, estimate_max_batch_size_per_gpu
from tomo2seg import callbacks as tomo2seg_callbacks
from tomo2seg import losses as tomo2seg_losses
from tomo2seg import schedule as tomo2seg_schedule
from tomo2seg import utils as tomo2seg_utils
from tomo2seg import slackme
from tomo2seg import train
from tomo2seg.train import Args, TrainingFinished, FailedToFindBatchSize
from tomo2seg import hosts as t2s_hosts


# # Args

logger.warning(Args.versions)


# [manual-input]

args = train.Args(
    script_name = "train-11-paper-tomo-modelstripping-00.py",
    volume_name = t2s_datasets.VOLUME_COMPOSITE_V1[0],
    volume_version = t2s_datasets.VOLUME_COMPOSITE_V1[1],
    labels_version = t2s_datasets.VOLUME_COMPOSITE_V1_LABELS_REFINED3,
    
    batch_size_per_gpu=5,

    random_state_seed = 42, 
#     runid = 1610978353,
    runid = None,
    host = None,  # find it with socket.hostname
)


# [manual-input]
crop_shape = (160, 160, 1)  # multiple of 16 (requirement of a 4-level u-net)
model_nclasses = 3

model_master_name = "paper-unet-2d"
model_version = f"f16-stripping-sepconv"
model_type = T2SModelType.d2

model_factory_function = modular_unet.u_net
model_factory_kwargs = {
    'convlayer': modular_unet.ConvLayer.conv2d_separable,
    'input_shape': crop_shape,
    'output_channels': model_nclasses,
    'nb_filters_0': 16,
    'depth': 3,
    'sigma_noise': 0.03,
    'updown_conv_sampling': True,
    'unet_block_kwargs': {
        'kernel_size': 3,
        'res': True,
        'batch_norm': True,
        'dropout': 0.10
    },
    'unet_down_kwargs': {'batchnorm': True},
    'unet_up_kwargs': {'batchnorm': True}
    
}

try:
    
    try:
        t2s_model

    except NameError:

        logger.info("Creating a T2SModel.")

        t2s_model = T2SModel(
            model_master_name,
            model_version,
            runid=args.runid,
            factory_function=model_factory_function,
            factory_kwargs=model_factory_kwargs,
        )

    else:
        logger.warning("The model is already defined. To create a new one: `del t2s_model`")

    finally:
        logger.info(f"t2s_model\n{dict2str(asdict(t2s_model))}")
        logger.info(f"{t2s_model.name=}")


    # # Setup




    logger.setLevel(logging.DEBUG)
    random_state = np.random.RandomState(args.random_state_seed)

    n_gpus = len(tf.config.list_physical_devices('GPU'))

    tf_version = tf.__version__
    logger.info(f"{tf_version=}")

    logger.info(
        f"Hostname: {args.host.hostname}\nNum GPUs Available: {n_gpus}\nThis should be:\n\t"
        + '\n\t'.join(['2 on R790-TOMO', '1 on akela', '1 on hathi', '1 on krilin'])
    )

    logger.debug(
        "physical GPU devices:\n\t"
        + "\n\t".join(map(str, tf.config.list_physical_devices('GPU'))) + "\n"
        + "logical GPU devices:\n\t"
        + "\n\t".join(map(str, tf.config.list_logical_devices('GPU')))
    )

    # xla auto-clustering optimization (see: https://www.tensorflow.org/xla#auto-clustering)
    # this seems to break the training
    tf.config.optimizer.set_jit(False)

    # get a distribution strategy to use both gpus (see https://www.tensorflow.org/guide/distributed_training)
    gpu_strategy = tf.distribute.MirroredStrategy()
    logger.debug(f"{gpu_strategy=}")

    logger.info(f"{dict2str(asdict(args.host))}")

    MAX_INTERNAL_NVOXELS = int(
        args.host.gpu_max_memory_factor * t2s_hosts.MAX_INTERNAL_NVOXELS
    )

    logger.info(f"{MAX_INTERNAL_NVOXELS=} ({humanize.intcomma(MAX_INTERNAL_NVOXELS)})")


    # # Model

    logger.info("Creating the Keras model.")

    if args.train_mode.is_continuation:

        logger.warning("Training continuation: a model will be loaded.")

        if args.train_mode == Args.TrainMode.continuation_from_latest_model:
            logger.info("Using the LATEST model to continue the training.")
            load_model_path = t2s_model.model_path

        elif args.train_mode == Args.TrainMode.continuation_from_autosaved_model:
            logger.info("Using the AUTOSAVED model to continue the training.")
            load_model_path = t2s_model.autosaved_model_path

        elif args.train_mode == Args.TrainMode.continuation_from_autosaved2_best_model:
            logger.info("Using the (best) AUTOSAVED2 model to continue the training.")
            load_model_path = t2s_model.autosaved2_best_model_path

        else:
            raise ValueError(f"{args.train_mode=}")

    elif (
        t2s_model.model_path.exists()
        or t2s_model.autosaved_model_path.exists()
        or (
            t2s_model.autosaved2_best_model_path is not None
            and t2s_model.autosaved2_best_model_path.exists()
        )
    ):
        logger.error(
            f"The model seems to already exist but this is not a continuation. Please, make sure the arguments are correct.")
        raise ValueError(
            f"{args.train_mode=} ==> {args.train_mode.is_continuation=} {t2s_model.name=}")

    elif args.train_mode == Args.TrainMode.from_scratch:
        logger.info(f"A new model will be instantiated!")

    else:
        raise NotImplementedError(f"{args.train_mode=}")


    with gpu_strategy.scope():

        if args.train_mode.is_continuation:

            assert load_model_path.exists(
            ), f"Inconsistent arguments {args.train_mode.is_continuation=} {load_model_path=} {load_model_path.exists()=}."

            logger.info(f"Loading model {load_model_path.name}")

            model = keras.models.load_model(str(load_model_path), compile=False)

            assert model.name == t2s_model.name, f"{model.name=} {t2s_model.name=}"

        else:

            logger.info(
                f"Instantiating a new model with model_factory_function={model_factory_function.__name__}.")

            model = model_factory_function(
                name=t2s_model.name,
                **model_factory_kwargs
            )

        logger.info("Compiling the model.")

        # [manual-input]
        # using the avg jaccard is dangerous if one of the classes is too
        # underrepresented because it's jaccard will be unstable
        # to be verified!
        loss = tomo2seg_losses.jaccard2_flat

    #     optimizer = optimizers.Adam(lr=.003)
        optimizer = AdaBeliefOptimizer(learning_rate=1e-3, epsilon=1e-14, rectify=True)

        metrics = []

        logger.debug(f"{loss=}")
        logger.debug(f"{optimizer=}")
        logger.debug(f"{metrics=}")

        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)





    if not args.train_mode.is_continuation:

        logger.info(f"Saving the model at {t2s_model.model_path=}.")
        model.save(t2s_model.model_path)

        logger.info(f"Writing the model summary at {t2s_model.summary_path=}.")

        with t2s_model.summary_path.open("w") as f:
            def print_to_txt(line):
                f.writelines([line + "\n"])
            model.summary(print_fn=print_to_txt, line_length=140)

        logger.info(
            f"Printing an image of the architecture at {t2s_model.architecture_plot_path=}.")

        utils.plot_model(model, show_shapes=True,
                         to_file=t2s_model.architecture_plot_path)





    # it is here because before this cell the folder doesnt exist yet
    add_file_handler(logger, t2s_model.train_log_path)

    # repeat it so that the log file saves this
    logger.info(f"args\n{dict2str(asdict(args))}")
    logger.info(f"{t2s_model.name=}")
    logger.info(f"t2s_model\n{dict2str(asdict(t2s_model))}")


    # # Data




    # Metadata/paths objects

    # Volume
    volume = Volume.with_check(
        name=args.volume_name, version=args.volume_version
    )

    logger.info(f"volume\n{dict2str(asdict(volume))}")

    assert volume.nclasses == model_nclasses, f"{model_nclasses=} {volume.nclasses=}"

    logger.info("Loading data from disk.")

    # Data
    voldata = file_utils.HST_read(
        str(volume.data_path),  # it doesn't accept paths...

        autoparse_filename=False,  # the file names are not properly formatted
        data_type=volume.metadata.dtype,
        dims=volume.metadata.dimensions,
        verbose=False,

    ) / volume.normalization_factor

    logger.debug(f"{voldata.shape=}")

    voldata_train = volume[args.partition_train].get_volume_partition(voldata)
    voldata_val = volume[args.partition_val].get_volume_partition(voldata)

    logger.debug(f"{voldata_train.shape=}")
    logger.debug(f"{voldata_val.shape=}")

    del voldata

    # Labels

    vollabels = file_utils.HST_read(
        str(volume.versioned_labels_path(args.labels_version)),

        autoparse_filename=False,
        data_type="uint8",
        dims=volume.metadata.dimensions,
        verbose=False,
    )

    logger.debug(f"{vollabels.shape=}")

    vollabels_train = volume[args.partition_train].get_volume_partition(vollabels)
    vollabels_val = volume[args.partition_val].get_volume_partition(vollabels)

    logger.debug(f"{vollabels_train.shape=}")
    logger.debug(f"{vollabels_val.shape=}")

    del vollabels


    # # Data crop sequences

    # ## Batch size




    logger.info(f"{args.batch_size_per_gpu=}")

    if args.batch_size_per_gpu is not None:
        batch_size_per_gpu = args.batch_size_per_gpu

    else: 
        batch_size_per_gpu = estimate_max_batch_size_per_gpu(
            model, 
            MAX_INTERNAL_NVOXELS,
            crop_shape,
        )
        logger.info(f"{batch_size_per_gpu=} (auto max)")

    batch_size = batch_size_per_gpu * max(1, n_gpus)

    logger.info(f"{n_gpus=} * {batch_size_per_gpu=} = {batch_size=}")


    # ## Common kwargs


    model_type_2_gt_type = {
        T2SModelType.d2: t2s_volseq.GT2D,
        T2SModelType.d2half: t2s_volseq.GT2D,
        T2SModelType.d3: t2s_volseq.GT3D,
    }

    gt_type = model_type_2_gt_type[model_type]

    logger.debug(f"{gt_type=}")

    vol_crop_seq_common_kwargs = dict(

        output_as_2d = (model_type == T2SModelType.d2),
        output_as_2halfd = (model_type == T2SModelType.d2half),
        labels = volume.metadata.labels,

        # [manual-input]
        debug__no_data_check=True,
    )

    logger.debug(f"{vol_crop_seq_common_kwargs=}")

    # ## Train

    data = voldata_train
    labels = vollabels_train

    volume_shape = data.shape

    crop_seq_train = VolumeCropSequence(

        data_volume=data,
        labels_volume=labels,

        batch_size=batch_size,

        meta_crop_generator = MetaCrop3DGenerator(

            volume_shape = volume_shape, 
            crop_shape = crop_shape,
            is_2halfd = (model_type == T2SModelType.d2half),

            x0y0z0_generator = (grid_pos_gen := t2s_volseq.UniformGridPosition.build_from_volume_crop_shapes(
                volume_shape = volume_shape, 
                crop_shape = crop_shape,
                random_state=RandomState(args.random_state_seed),
            )),

            gt_field = t2s_volseq.GTUniformEverywhere(
                gt_type = gt_type,
                grid_position_generator = grid_pos_gen,
                random_state = RandomState(args.random_state_seed),
            ),

            et_field = t2s_volseq.ET3DConstantEverywhere.build_no_displacement(grid_position_generator=grid_pos_gen),

            vs_field = t2s_volseq.VSConstantEverywhere.build_no_shift(grid_position_generator=grid_pos_gen)

        ),

        # this volume cropper only returns random crops,
        # so the number of crops per epoch/batch is w/e i want
        epoch_size=10,

        **vol_crop_seq_common_kwargs,

        meta_crops_hist_path=t2s_model.train_metacrop_history_path,
    )


    # ## Val

    data = voldata_val
    labels = vollabels_val

    volume_shape = data.shape

    grid_pos_gen = SequentialGridPosition.build_min_overlap(
        volume_shape=volume_shape,
        crop_shape=crop_shape,

        # [manual-input]
        # reduce the total number of crops
    #         n_steps_x=11,
    #         n_steps_y=11,
        n_steps_z=10,
    )

    crop_seq_val = VolumeCropSequence(

        data_volume=data,
        labels_volume=labels,

        batch_size=batch_size,

        # go through all the crops in validation
        epoch_size=len(grid_pos_gen),

        # data augmentation
        meta_crop_generator=MetaCrop3DGenerator.build_no_augmentation(
            grid_pos_gen = grid_pos_gen,
            volume_shape = volume_shape,
            crop_shape = crop_shape,
            common_random_state_seed = args.random_state_seed,
            gt_type = gt_type,
            is_2halfd = (model_type == T2SModelType.d2half),
        ),

        meta_crops_hist_path=t2s_model.val_metacrop_history_path,

        **vol_crop_seq_common_kwargs,
    )


    # # Callbacks

    # this is important because sometimes i update things in the notebook
    # so i need to make sure that the objects in the history cb are updated
    try:
        history_cb

    except NameError:

        logger.info("Creating a new history callback.")

        history_cb = tomo2seg_callbacks.History(
            optimizer=model.optimizer,
            crop_seq_train=crop_seq_train,
            crop_seq_val=crop_seq_val,
            backup=1,
            csv_path=t2s_model.history_path,
        )

    else:

        logger.warning("The history callback already exists!")

        history_df = history_cb.dataframe

        try:
            history_df_temp = pd.read_csv(t2s_model.history_path)
            # keep the longest one
            history_df = history_df if history_df.shape[0] >= history_df_temp.shape[0] else history_df_temp
            del history_df_temp

        except FileNotFoundError:
            logger.info("History hasn't been saved yet.")

        except pd.errors.EmptyDataError:
            logger.info("History hasn't been saved yet.")

    finally:
        # make sure the correct objects are linked
        history_cb.optimizer = model.optimizer
        history_cb.crop_seq_train = crop_seq_train
        history_cb.crop_seq_val = crop_seq_val

    logger.debug(f"{history_cb=}")
    logger.debug(f"{history_cb.dataframe.index.size=}")
    logger.debug(f"{history_cb.last_epoch=}")

    history_plot_cb = tomo2seg_callbacks.HistoryPlot(
        history_callback=history_cb,
        save_path=t2s_model.train_history_plot_wip_path
    )
    logger.debug(f"{history_plot_cb=}")

    # [manual-input]
    n_epochs = 200

    initial_lr = optimizer.lr.numpy()

    lr_schedule_cb = keras_callbacks.LearningRateScheduler(
        schedule=(
    #         schedule := tomo2seg_schedule.get_schedule00()
            schedule := tomo2seg_schedule.LinSpaceSchedule(
                offset_epoch=0, wait=100, start=initial_lr, stop=initial_lr / 10, n_between=100
            )
        ),
        verbose=2,
    )

    # todo plot schedule

    logger.info(f"{lr_schedule_cb.schedule.range=}")

    callbacks = [
        keras_callbacks.TerminateOnNaN(),
        keras_callbacks.ModelCheckpoint(
            t2s_model.autosaved2_model_path_str,
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            mode="min",
        ),

        history_cb,
        history_plot_cb,
        lr_schedule_cb,
    ]

    try:
        early_stop_cb

    except NameError:
        pass

    else:
        callbacks.append(early_stop_cb)

    for cb in callbacks:
        logger.debug(f"using callback {cb.__class__.__name__}")





    def fit():
    #     raise NotImplementedError("I have to automate the logic of the initial epoch...")

        model.fit(
            # data sequences
            x=crop_seq_train,
            validation_data=crop_seq_val,

            # [manual-input]
            # epochs
            initial_epoch=0 if history_cb.last_epoch == 0 else history_cb.last_epoch + 1,
            epochs=n_epochs,
    #         epochs=history_cb.last_epoch + 1 + n_epochs,

            # others
            callbacks=callbacks,
            verbose=2,

            # todo change the volume sequence to dinamically load the volume
            # because it would allow me to pass just a path string therefore
            # making it serializible ==> i will be able to multithread (:
            use_multiprocessing=False,
        )
        raise TrainingFinished()


    while True:

        try:
            fit()

        except TrainingFinished:
            slack.notify_finished()
            break

        except Exception as ex:

            logger.exception(ex)

            if args.batch_size_mode == Args.BatchSizeMode.try_max_and_fail:
                raise ex

            batch_size -= n_gpus
            logger.warning(f"reduced {batch_size=}")

            if batch_size < n_gpus:
                raise FailedToFindBatchSize

            crop_seq_train.batch_size = batch_size
            crop_seq_val.batch_size = batch_size


    # # History




    fig, axs = plt.subplots(nrows := 2, ncols := 1, figsize=(
        2.5 * (sz := 5), nrows * sz), dpi=100)
    fig.set_tight_layout(True)

    hist_display = viz.TrainingHistoryDisplay(
        history_cb.history,
        model_name=t2s_model.name,
        loss_name=model.loss.__name__,
        x_axis_mode=(
            "epoch", "batch", "crop", "voxel", "time",
        ),
    ).plot(
        axs,
        with_lr=True,
        metrics=(
            "loss",
        ),
    )

    axs[0].set_yscale("log")
    axs[-1].set_yscale("log")

    viz.mark_min_values(hist_display.axs_metrics_[0], hist_display.plots_["loss"][0])
    viz.mark_min_values(hist_display.axs_metrics_[0], hist_display.plots_["val_loss"][0], txt_kwargs=dict(rotation=0))

    hist_display.fig_.savefig(
        t2s_model.model_path / (hist_display.title + ".png"),
        format='png',
    )





    history_cb.dataframe.to_csv(history_cb.csv_path, index=True)





    model.save(t2s_model.model_path)


    # # End




    slack.notify(f"script `{args.script_name}` on `{args.host.hostname}` finished!")

except Exception as ex:
    slack.notify_exception(ex, args.host.hostname)






