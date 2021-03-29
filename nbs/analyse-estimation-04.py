# coding: utf-8

# # Imports
import copy
import functools
import gc
import itertools
import logging
import operator
import os
import pathlib
import re
import socket
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum
from functools import partial
from pathlib import Path
from pprint import PrettyPrinter, pprint
from typing import *

import humanize
import matplotlib
import numpy as np
import pandas as pd
import scipy as sp
import tensorflow as tf
import yaml
from matplotlib import cm, patches, pyplot as plt
from numpy import ndarray
from numpy.random import RandomState
from progressbar import progressbar as pbar
from pymicro.file import file_utils
from sklearn import metrics, metrics as met, model_selection, preprocessing
from skimage import measure as skimage_measure
import tabulate
from tensorflow import keras
from tensorflow.keras import (
    callbacks as keras_callbacks,
    layers,
    losses,
    metrics as keras_metrics,
    optimizers,
    utils,
)
from tqdm import tqdm
from yaml import YAMLObject

from tomo2seg import (
    analyse as t2s_analyse,
    callbacks as tomo2seg_callbacks,
    data as tomo2seg_data,
    hosts,
    losses as tomo2seg_losses,
    schedule as tomo2seg_schedule,
    slack,
    slackme,
    utils as tomo2seg_utils,
    viz as t2s_viz,
    volume_sequence,
)
from tomo2seg.data import EstimationVolume, Volume
from tomo2seg.logger import add_file_handler, dict2str, logger
from tomo2seg.model import Model as Tomo2SegModel
from tomo2seg.analyse_pred import AnalysePredMetaArgs as MetaArgs
from tomo2seg.analyse_pred import AnalysePredOuputs as Outputs
from tomo2seg.analyse_pred import AnalysePredOpts as Opts
from tomo2seg import hosts as t2s_hosts
from tomo2seg import datasets as t2s_datasets
from tomo2seg import analyse_gt, analyse_pred, analyse


# # MetaArgs

# [manual-input]
meta_args = MetaArgs(
    
    script_name = "analyse-estimation-04.py",
    
    volume_name = t2s_datasets.VOLUME_COMPOSITE_V1[0],
    volume_version = t2s_datasets.VOLUME_COMPOSITE_V1[1],
    labels_version = t2s_datasets.VOLUME_COMPOSITE_V1_LABELS_REFINED3,
    
    estimation_volume_fullname = sys.argv[1],
#     estimation_volume_fullname = (
#         "vol=PA66GF30.v1.set=test.model=paper-unet-2d.full-f16.fold000.1611-743-205.runid=1611-789-693"
#     ),
    
    opts = Opts(
        compute = Opts.Compute(
            error_volume = True,

            roc_curve = False,
            multiclass_roc_auc = False,

            error_blobs_2d_props = True,
            error_blobs_3d_props = False,

            adjacent_layers_correlation = True,
        ),
        save = Opts.Save(
            confusion_volume = False,
            error_volume = True,
            
            error_blobs_2d_props = False,
            error_blobs_3d_props = False
        ),
    ),
    
    host=None,  # None = auto
    runid=None,  # None = auto
    random_state_seed=42,  # None = auto
)

try: 
    # # `tomo2seg` objects 

    volume = Volume.with_check(
        name=meta_args.volume_name, 
        version=meta_args.volume_version,
    )

    estimation_volume = EstimationVolume.from_fullname(
        meta_args.estimation_volume_fullname,
    )

    assert estimation_volume.volume_fullname == volume.fullname

    if estimation_volume.partition is None:
        raise NotImplementedError(f"{estimation_volume.partition=}")

    # todo replace me by a similar structure inside the outputs object
    estimation_volume["analyse-pred-meta_args"] = asdict(meta_args)

    # # Args

    log_filepath = estimation_volume.analyse_exec_log_path

    random_state_seed = meta_args.random_state_seed
    random_state = np.random.RandomState(random_state_seed)

    runid = meta_args.runid

    script_name = meta_args.script_name
    hostname = meta_args.host.hostname

    opts = copy.deepcopy(meta_args.opts)

    data_path = str(volume.data_path)
    data_meta = dict(
        dtype = volume.metadata.dtype,
        dims = volume.metadata.dimensions,
    )

    labels_path = str(volume.versioned_labels_path(meta_args.labels_version))
    preds_path = str(estimation_volume.predictions_path)
    probas_path = str(estimation_volume.probabilities_path)

    partition = estimation_volume.partition
    partition_dims = (
        estimation_volume.partition.shape 
        if estimation_volume.partition is not None else 
        volume.metadata.dimensions
    )
    partition_slice = analyse_gt.partition2slice(partition) 

    # float16 instead of 64 to save memory
    proba_dtype = np.float16

    cv_dtype = np.int16

    labels_idx = volume.metadata.labels
    labels_names = [volume.metadata.labels_names[idx] for idx in labels_idx]
    labels_idx_name = dict(zip(labels_idx, labels_names))
    n_classes = len(labels_idx)

    # parallel_nprocs = host.analyse_parallel_nprocs
    parallel_nprocs = None  # use it all

    # todo move me to the estim volume obs
    outputs_dir = estimation_volume.dir / "pred-analysis"  
    outputs_dir.mkdir(exist_ok=True)


    # # Outputs




    outputs = Outputs(outputs_dir)


    # # Setup




    logger.setLevel(logging.DEBUG)
    add_file_handler(logger, log_filepath)


    # # Log stuff




    logger.info(
        f"{volume.__class__.__name__}{dict2str(asdict(volume))}"
    )
    logger.info(
        f"{estimation_volume.__class__.__name__}{dict2str(asdict(estimation_volume))}"
    )
    logger.info(
        f"{opts.__class__.__name__}{dict2str(asdict(opts))}"
    )


    # # Exec

    # ## Load data




    logger.info("Loading data from disk.")
    data_volume = file_utils.HST_read(
        data_path,  # it doesn't accept paths...
        autoparse_filename=False,  # the file names are not properly formatted
        data_type=data_meta["dtype"],
        dims=data_meta["dims"],
        verbose=False,
    )[partition_slice]
    logger.debug(f"{data_volume.shape=}")

    logger.info("Loading labels from disk.")
    labels_volume = file_utils.HST_read(
        labels_path,  # it doesn't accept paths...
        autoparse_filename=False,  # the file names are not properly formatted
        data_type="uint8",
        dims=data_meta["dims"],
        verbose=False,
    )[partition_slice]
    logger.debug(f"{labels_volume.shape=}")





    logger.info("Loading predictions from disk.")
    preds_volume = file_utils.HST_read(
        preds_path,  # it doesn't accept paths...
        autoparse_filename=False,  # the file names are not properly formatted
        data_type="uint8",
        dims=partition_dims,
        verbose=False,
    )
    logger.debug(f"{preds_volume.shape=}")





    logger.info("Loading probabilities from disk.")
    probas_volume = np.load(probas_path).astype(proba_dtype)
    logger.debug(f"{probas_volume.shape=}")


    # ## confusion volume

    # ### [compute] confusion volume




    logger.info("Computing confusion volume.")

    cv_encoding, cv_encoding_inv = analyse_pred.get_conf_vol_encoding(labels_idx)

    logger.debug(f"cv_encoding\n{dict2str(cv_encoding)}")
    estimation_volume["cv_encoding"] = cv_encoding

    logger.debug(f"cv_encoding_inv\n{dict2str(cv_encoding_inv)}")
    estimation_volume["cv_encoding_inv"] = cv_encoding_inv

    # 10000 is an impossible encoding
    conf_vol = np.full_like(labels_volume, 10000, dtype=cv_dtype)

    for (gt_idx, pred_idx), encoded_value in pbar(
        cv_encoding.items(),
        max_value=len(cv_encoding)
    ):
        conf_vol[
            (labels_volume == gt_idx) & (preds_volume == pred_idx)
        ] = encoded_value





    assert np.all(conf_vol != 10000), "10000 is an impossible encoding"


    # ### [save] confusion volume




    logger.info(f"Saving confusion volume.")

    if opts.save.confusion_volume:
        file_utils.HST_write(conf_vol, str(outputs.confusion_volume))   
        logger.info("done")
    else: 
        logger.info("skipped")


    # ## error volume

    # ### [compute] error volume




    logger.info("Computing error volume.")

    if opts.compute.error_volume:

        error_volume = np.full_like(labels_volume, False, dtype=bool)

        for label_idx in pbar(labels_idx):

            encoded_value = cv_encoding[(label_idx, label_idx)]

            error_volume |= (conf_vol == encoded_value)

        error_volume = ~error_volume

        logger.info("done")

    else: 
        logger.info("skipped")


    # ### [save] error volume 




    logger.info(f"Saving error volume.")

    if opts.save.error_volume:
        file_utils.HST_write(error_volume, str(outputs.error_volume_path))    
        logger.info("done")
    else:
        logger.info("skipped")


    # ## confusion matrix

    # ### [compute] confusion matrix




    logger.info("Computing confusion matrix.")

    max_encoded_val = max(cv_encoding.values())

    logger.debug(f"{max_encoded_val=}")

    # cm = confusion matrix
    cm_encoded_counts = np.bincount(conf_vol.ravel(), minlength=max_encoded_val + 1)

    cm_counts = {}

    for gt_pred_indices, enc_val in cv_encoding.items():

        cm_counts[gt_pred_indices] = cm_encoded_counts[enc_val]

    conf_matrix = [
        [
            cm_counts[(gt_idx, pred_idx)]
            for pred_idx in labels_idx
        ]
        for gt_idx in labels_idx
    ]

    conf_matrix = np.array(conf_matrix)

    try:

        assert (ncorrect_error_volume := (~error_volume).sum()) == (ncorrect_conf_matrix := conf_matrix.diagonal().sum()), (
            f"{ncorrect_error_volume=} {ncorrect_conf_matrix=}"
        )

    except NameError as ex:

        if ex.args[0] != "name 'error_volume' is not defined":
            raise ex

        # never mind...

    logger.info("done")


    # ### [save] confusion matrix




    logger.info(f"Saving confusion matrix.")
    estimation_volume["confusion_matrix_dtype"] = str(conf_matrix.dtype)
    np.save(outputs.confusion_matrix, conf_matrix)
    logger.info("done")


    # ## [compute][save] roc curve




    logger.info("Computing and saving ROC curves")

    if opts.compute.roc_curve:

        roc_dfs = []

        for label_idx in pbar(labels_idx):

            logger.debug(f"computing roc curve {label_idx=}")

            fpr, tpr, th = metrics.roc_curve(
                labels_volume.ravel(), 
                probas_volume[:, :, :, label_idx].ravel(), 

                pos_label=label_idx,
                drop_intermediate=True
            )

            roc_df = pd.DataFrame(
                data={
                    "fpr": fpr,
                    "tpr": tpr,
                    "th": th,
                }
            ).T

            logger.debug(f"{label_idx=} {roc_df.shape=}")

            roc_dfs.append(roc_df)

            roc_path = outputs.roc_curve(label_idx)

            logger.debug(f"saving roc curve {label_idx=} at {roc_path=}")

            roc_df.to_csv(
                roc_path,
                header=True,
                index=True,
            )

        logger.info("done")

    else: 
        logger.info("skipped")


    # # [compute] multi-class roc-auc




    logger.info("Computing the multiclass ROC curves")

    if opts.compute.multiclass_roc_auc:

        raveled_probas = probas_volume.reshape(-1, n_classes)
        raveled_probas = raveled_probas / raveled_probas.sum(axis=-1, keepdims=True)  # more numerically precise...

        multiclass_roc_auc_macro_ovr = t2s_analyse.multiclass_roc_auc_score(
            y_true=labels_volume.ravel(),
            y_score=raveled_probas,
            average="macro",
            multi_class="ovr",
            labels=labels_idx,
        )

        logger.debug(f"{multiclass_roc_auc_macro_ovr=}")

        multiclass_roc_auc_macro_ovo = t2s_analyse.multiclass_roc_auc_score(
            y_true=labels_volume.ravel(),
            y_score=raveled_probas,
            average="macro",
            multi_class="ovo",
            labels=labels_idx,
        )

        logger.debug(f"{multiclass_roc_auc_macro_ovo=}")

        logger.info("done")

    else: 
        logger.info("skipped")


    # ## 2d error blobs

    # ### [compute] 2d error blobs




    logger.info("Computing 2d error blobs in the 3 directions.")

    if opts.compute.error_blobs_2d_props:
        error_2dblobs_props = analyse.get_2d_blob_props(
            label_volume=error_volume,
            data_volume=data_volume,
            parallel_nprocs=parallel_nprocs,
        )
        logger.info("done")

    else: 
        logger.info("skipped")


    # ### [save] 2d error blobs




    logger.info("Saving 2d error blobs.")

    if opts.save.error_blobs_2d_props:
        error_2dblobs_props.to_csv(outputs.error_2dblobs_props, index=False)
        logger.info("done")

    else: 
        logger.info("skipped") 


    # ## 3d error blobs

    # ### [compute] 3d error blobs

    # ### [save] 3d error blobs

    # ## derived computations

    # ### classification report




    rocs = None if not opts.compute.roc_curve else tuple(
        {
            "tpr": roc_df.loc["tpr"].values,
            "fpr": roc_df.loc["fpr"].values,
        }
        for roc_df in roc_dfs
    )

    report_dict = t2s_analyse.get_classification_report(
        cm=conf_matrix,
        rocs=rocs,
    )

    if opts.compute.multiclass_roc_auc:
        report_dict["macro"]["multiclass-roc-auc-ovr"] = float(multiclass_roc_auc_macro_ovr)
        report_dict["macro"]["multiclass-roc-auc-ovo"] = float(multiclass_roc_auc_macro_ovo)





    for idx, name in labels_idx_name.items():
        report_dict[name] = report_dict[idx]
        del report_dict[idx]





    logger.info(f"Saving classification report.")

    yaml_dump = functools.partial(
        yaml.dump,
        default_flow_style=False, 
        indent=4, 
        sort_keys=False
    )

    with outputs.classification_report_exact.open('w') as f:
        yaml_dump(report_dict, f)

    with outputs.classification_report_human.open('w') as f:
        humanized_report_str = yaml_dump(
            report_dict, 
            Dumper=t2s_analyse.ClassifReportHumandDumper,
        )
        f.write(humanized_report_str)


    # # plots

    # ## classification report (table)




    (
        df,
        table_human_simple,
        table_human_detail
    ) = analyse_pred.report2table(report_dict, labels_names)





    df.to_csv(
        outputs.classification_report_table_csv, 
        header=True,
        index=True,
    )

    table_str = tabulate.tabulate(table_human_simple, headers=df.columns.values.tolist())

    with outputs.classification_report_table_human.open("w") as f:
        f.write(table_str)

    table_str = tabulate.tabulate(table_human_detail, headers=df.columns.values.tolist())

    with outputs.classification_report_table_exact.open("w") as f:
        f.write(table_str)



    # ## confusion matrix




    estimation_volume_alias = estimation_volume.fullname  # todo move me to args





    fig, axs = plt.subplots(
        n_rows := 2, 
        n_cols := 2, 
        figsize=(n_cols * (sz := 4), n_rows * sz), 
        dpi=(dpi := 100),
        gridspec_kw=dict(wspace=sz/30),
    )

    cm_display = t2s_viz.ConfusionMatrixDisplay(
        cm_normalized := conf_matrix, 
        display_labels=labels_names,
    ).plot(
        values_format=None, 
        cmap=cm.inferno, 
        ax=axs[0, 0],
        cmap_vmax=int(conf_matrix.max()),
    )

    cm_display.ax_.set_title("counts")

    cm_display = t2s_viz.ConfusionMatrixDisplay(
        cm_normalized := conf_matrix / conf_matrix.sum(), 
        display_labels=labels_names,
    ).plot(
        values_format='.2%', 
        cmap=cm.inferno, 
        ax=axs[0, 1]
    )

    cm_display.ax_.set_title("normalized (global)")

    cm_display = t2s_viz.ConfusionMatrixDisplay(
        cm_true_label_normalized := conf_matrix / conf_matrix.sum(axis=1).reshape(-1, 1), 
        display_labels=labels_names,
    ).plot(
        values_format='.1%', 
        cmap=cm.inferno, 
        ax=axs[1, 0],
    )
    cm_display.ax_.set_title("norm. by *GT* (line)\ndiagonal = recall")

    cm_display = t2s_viz.ConfusionMatrixDisplay(
        cm_predicted_label_normalized := conf_matrix / conf_matrix.sum(axis=0).reshape(1, -1), 
        display_labels=labels_names,
    ).plot(
        values_format='.1%', 
        cmap=cm.inferno, 
        ax=axs[1, 1],
    )

    cm_display.ax_.set_title("norm. by *PRED* (column)\ndiagonal = precision")

    fig.suptitle(f"Confusion matrices {estimation_volume_alias}");

    fig.savefig(
        fname=outputs.confusion_matrices_plot,
        format="png",
        dpi=dpi,
    )


    # ## roc curve




    logger.info("plotting roc curves")

    if opts.compute.roc_curve:

        fig, axs = plt.subplots(
            n_rows := 1, 
            n_cols := 2, 
            figsize=(n_cols * (sz := 7), n_rows * sz), 
            dpi=(dpi := 130),
        )

        zoom = np.array(((0, .15), (.85, 1)))

        fig.suptitle("Per class ROC curves")

        ax_full, ax_zoom = axs[0], axs[1]
        ax_full.set_title("Full curve range [0, 1] x [0, 1]")
        ax_full.set_xlim(0, 1)
        ax_full.set_ylim(0, 1)

        ax_zoom.set_title(f"Zoom on [{zoom[0, 0]}, {zoom[0, 1]}] x [{zoom[1, 0]}, {zoom[1, 1]}]")
        ax_zoom.set_xlim(*zoom[0])
        ax_zoom.set_ylim(*zoom[1])

        for label_idx, roc_df in zip(labels_idx, roc_dfs):

            fpr = roc_df.loc["fpr"].values
            tpr = roc_df.loc["tpr"].values

            roc_display = metrics.RocCurveDisplay(
                fpr=fpr, 
                tpr=tpr, 
                estimator_name=f"{label_idx}",
            )

            for ax in axs:
                roc_display.plot(ax=ax)  

        max_label_name_length = max(*map(len, labels_names))

        for label_idx, roc_df in zip(labels_idx, roc_dfs):

            fpr = roc_df.loc["fpr"].values
            tpr = roc_df.loc["tpr"].values

            label_name = labels_names[label_idx]

            ax_full.get_legend().texts[label_idx].set_text(
                label_name.ljust(max_label_name_length) +
                f"AUC={report_dict[label_name]['roc-auc']:.2%}"
            )

        ax_zoom.legend_ = None

        fig.savefig(outputs.roc_plot, format='png')

        logger.info("done")

    else:
        logger.info("skipped")


    # ## volumetric fraction




    class_proportion_gt = conf_matrix.sum(axis=1)
    class_proportion_pred = conf_matrix.sum(axis=0)





    partition_alias = estimation_volume.partition.alias  # todo move me to args
    volume_name = volume.fullname  # todo move me to args





    fig, axs = plt.subplots(
        nrows := 1, ncols := 2, 
        figsize=(ncols * (sz := 7), nrows * sz), 
        dpi=(dpi := 90), 
        gridspec_kw=dict(wspace=sz/16, hspace=sz/12)
    )

    common_kwargs = dict(
        barh_kwargs=dict(
            height=.6,
        ),
        count_fmt_func=lambda c: f"{humanize.intword(c)}",
        perc_fmt_func=lambda p: f"{p:.1%}",
    )

    display_gt = t2s_viz.ClassImbalanceDisplay(
        volume_name=f"ground truth",
        labels_idx=labels_idx,
        labels_names=labels_names,
        labels_counts=class_proportion_gt.tolist(),
    ).plot(ax=axs[0], **common_kwargs)

    display_pred = t2s_viz.ClassImbalanceDisplay(
        volume_name=f"prediction",
        labels_idx=labels_idx,
        labels_names=labels_names,
        labels_counts=class_proportion_pred.tolist(),
    ).plot(ax=axs[1], **common_kwargs)

    fig.suptitle(f"{volume_name} ({partition_alias} set): volumetric fraction comparison.")

    fig.savefig(
        fname=outputs.volumetric_fraction_plot,
        format="png",
    )


    # # Physical metrics




    # - voxel size
    # - volume size 
    # - fiber length
    # - fiber diameter
    # - porosity diameter
    # - fraction volumique


    # ## adjacent layers correlation

    # ### [compute] adjacent layers correlation




    @dataclass
    class AdjacentLayerCorrelation:

        class Dataset(Enum):
            gt = 0
            pred = 1

        dataset: str  # 'gt'/'pred'
        axis: int
        label: Optional[int]  

        values: List[int] = field(repr=False)

    correlations = [
        AdjacentLayerCorrelation(
            dataset=AdjacentLayerCorrelation.Dataset.gt,
            axis = axis,
            label = label,
            values = analyse_gt.adjacent_layers_correlation(
                labels=labels_volume,
                axis=axis,
                nslices=1,
                correlation_func=partial(analyse_gt.jaccard, label=label)
            )
        )
        for axis, label in pbar(list(itertools.product(
            list(range(3)),
            [None] + list(labels_idx),
        )))
    ] + [
        AdjacentLayerCorrelation(
            dataset=AdjacentLayerCorrelation.Dataset.pred,
            axis = axis,
            label = label,
            values = analyse_gt.adjacent_layers_correlation(
                labels=labels_volume,
                axis=axis,
                nslices=1,
                correlation_func=partial(analyse_gt.jaccard, label=label)
            )
        )
        for axis, label in pbar(list(itertools.product(
            list(range(3)),
            [None] + list(labels_idx),
        )))
    ]


    # ### [save] adjacent layers correlation




    logger.info("Saving adjacent layers correlation series.")

    for corr in pbar(correlations):

        if corr.dataset != AdjacentLayerCorrelation.Dataset.pred:
            continue

        filepath = outputs.layers_correlation(
            axis=corr.axis,
            label=corr.label,
        )

        np.save(filepath, corr.values)


    # ### [plot] adjacent layers correlation
    # todo make this a display



    fig, axs = plt.subplots(
        nrows := n_classes + 1,
        ncols := 3,
        figsize = (
            ncols * (sz := 5),
            nrows * sz,
        ),
        dpi = 100,
    )


    def corr2ax(corr: AdjacentLayerCorrelation):
        return axs[
            corr.label if corr.label is not None else -1, 
            corr.axis
        ]


    for corr in correlations:
        ax = corr2ax(corr)
        ax.plot(
            corr.values,
            label=f"{corr.dataset.name}",
            linewidth=1,
            linestyle='-' if corr.dataset == AdjacentLayerCorrelation.Dataset.gt else ":",
    #         linestyle='-',
    #         linestyle=':',
        )

    for ax in axs.ravel():
        ax.set_ylim(0, 1)
        ax.legend()

    for axis in range(3):
        for label in list(range(n_classes)) + [None]:
            ax_ = axs[label if label is not None else -1, axis]
            ax_.set_title(
                f"{axis=} label={label if label is not None else 'all'}"
            )

    fig.suptitle(f"{volume_name} adjacent layer correlation")

    fig.savefig(fname=outputs.layers_correlation_plot, format="png")
    plt.close();


    # # [todo] measure the classification metrics layerwise in all the axes

    # # Notable slices




    logger.info(f"Finding notable slices.")

    if opts.compute.error_blobs_2d_props:

        MIN_ERROR_BLOB_AREA = 1

        logger.info(f'filtering error blobs < {MIN_ERROR_BLOB_AREA=}')

        logger.debug(f"before {(nblobs := error_2dblobs_props.shape[0])=} ({humanize.intcomma(nblobs)})")

        error_2dblobs_props = error_2dblobs_props[error_2dblobs_props.area > MIN_ERROR_BLOB_AREA]

        logger.debug(f"after {(nblobs := error_2dblobs_props.shape[0])=} ({humanize.intcomma(nblobs)})")

        if error_2dblobs_props.index.name != "normal_axis":

            error_2dblobs_props = error_2dblobs_props.reset_index().set_index(["normal_axis"])

        notable_slices = {}

        add_notable_slices = functools.partial(
            t2s_analyse.add_notable_slices,
            notable_slices=notable_slices,
            error_2dblobs_props=error_2dblobs_props,
        )

        add_notable_slices_blobwise = functools.partial(
            t2s_analyse.add_notable_slices_blobwise,
            notable_slices=notable_slices,
            error_2dblobs_props=error_2dblobs_props,
        )

        add_notable_slices_blobwise(t2s_analyse.max_area)

        add_notable_slices_blobwise(partial(t2s_analyse.max_bbox_shape, dim=1), axes=(0,))
        add_notable_slices_blobwise(partial(t2s_analyse.max_bbox_shape, dim=2), axes=(0,))
        add_notable_slices_blobwise(partial(t2s_analyse.max_bbox_shape, dim=0), axes=(1,))
        add_notable_slices_blobwise(partial(t2s_analyse.max_bbox_shape, dim=2), axes=(1,))
        add_notable_slices_blobwise(partial(t2s_analyse.max_bbox_shape, dim=0), axes=(2,))
        add_notable_slices_blobwise(partial(t2s_analyse.max_bbox_shape, dim=1), axes=(2,))

        add_notable_slices_blobwise(t2s_analyse.max_major_axis_length)

        add_notable_slices_blobwise(t2s_analyse.max_minor_axis_length)

        add_notable_slices(t2s_analyse.max_error_area)

        add_notable_slices(t2s_analyse.max_error_blob_avg_area)

        with outputs.notable_slices_yaml.open('w') as f:
            yaml_dump(notable_slices, f)

        logger.info("done")

    else: 
        logger.info("skipped")


    # # Plot notable slices 




    logger.info("plotting notable slices")

    for name, obj in notable_slices.items():

        logger.debug(f"plotting {name}")

        slice_ = 3 * [slice(None, None, None)]
        slice_[obj["normal_axis"]] = slice(obj["slice_idx"], obj["slice_idx"] + 1, None)
        slice_ = tuple(slice_)

        slice_data = data_volume[slice_].squeeze(obj["normal_axis"])
        slice_pred = preds_volume[slice_].squeeze(obj["normal_axis"])
        slice_err = error_volume[slice_].squeeze(obj["normal_axis"])

        display_pred = t2s_viz.SliceDataPredictionDisplay(
            slice_data=slice_data,
            slice_prediction=slice_pred,
            slice_name=name,  # todo: make each slice have some semantic name ("max area = 420")
            n_classes=n_classes,
        )

        display_err = t2s_viz.SliceDataPredictionDisplay(
            slice_data=slice_data,
            slice_prediction=slice_err,
            slice_name=name,  # todo: make each slice have some semantic name ("max area = 420")
            n_classes=n_classes,
        )

        fig, axs = plt.subplots(2, 2, figsize=(sz := 15, 2 * sz), dpi=120)
        fig.set_tight_layout(True)

        display_pred.plot(axs[0], data_imshow_kwargs=dict(vmin=0, vmax=255))
        display_err.plot(axs[1], data_imshow_kwargs=dict(vmin=0, vmax=255))

        fig.savefig(
            fname = outputs.notable_slices_plot(name),
            format="png",
            metadata=display_pred.metadata,
        )       
        plt.close()


    # # End
    slack.notify(f"notebook `{script_name}` finished in `{hostname}`!")

except Exception as ex:
    
    slack.notify_exception(ex, hostname)






