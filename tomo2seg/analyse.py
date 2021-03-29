from functools import partial
from multiprocessing import Pool
from typing import Callable, Dict, List, Optional, Tuple, Union

import humanize
import numpy as np
import pandas as pd
import skimage
import skimage.measure
import yaml
from numpy import ndarray
from pandas import DataFrame
from progressbar import progressbar

# bins = np.linspace(*hist_range, n_bins).astype(int).tolist()
from sklearn import metrics
from tomo2seg.logger import logger

# hist_bins = np.linspace(0, 256, 257).astype(int)
# data_hist, _ = np.histogram(
#     data_volume.ravel(),
#     bins=hist_bins,
#     density=False,
# )
# labels_volume_raveled_t = tf.convert_to_tensor(
#     labels_volume.ravel(), dtype=tf.int8
# )
# data_hists_per_label = []
# data_hists_per_label_global_prop = []
# n_voxels = np.sum(labels_counts)
# for label_idx in labels_idx:
#     logger.debug(f"Computing histogram for {label_idx=}")
#     label_data_hist_t = tf.histogram_fixed_width(
#         values=data_volume_raveled_t[labels_volume_raveled_t == label_idx],
#         value_range=tf.constant(hist_range, dtype=data_volume_raveled_t.dtype),
#         nbins=n_bins,
#         dtype=data_volume_raveled_t.dtype,
#         name=f"{volume.fullname}.data-histogram"
#     )
#     data_hists_per_label.append(
#         (label_data_hist_t / tf.math.reduce_sum(label_data_hist_t)).numpy().tolist()
#     )
#     data_hists_per_label_global_prop.append(
#         (label_data_hist_t.numpy() / n_voxels).tolist()
#     )


def get_classification_report(
    cm: ndarray, rocs: Optional[Tuple[Dict[str, ndarray]]] = None
) -> dict:

    assert cm.dtype == np.int64, f"{cm.dtype=}"
    assert len(cm.shape) == 2, f"{cm.shape=}"
    assert cm.shape[0] == cm.shape[1], f"{cm.shape=}"

    nlabels = cm.shape[0]

    if rocs is not None:

        assert len(rocs) == nlabels, f"{len(rocs)=} {nlabels=}"

        for label_idx, roc_dict in enumerate(rocs):
            assert isinstance(roc_dict, dict), f"{label_idx=} {type(roc_dict)=}"
            assert "tpr" in roc_dict, f"{label_idx=} {roc_dict.keys()=}"
            assert "fpr" in roc_dict, f"{label_idx=} {roc_dict.keys()=}"
            tpr = roc_dict["tpr"]
            fpr = roc_dict["fpr"]
            assert isinstance(tpr, ndarray), f"{label_idx=} {type(roc_dict['tpr'])=}"
            assert isinstance(fpr, ndarray), f"{label_idx=} {type(roc_dict['fpr'])=}"
            assert len(tpr.shape) == 1, f"{label_idx=} {tpr.shape=}"
            assert len(fpr.shape) == 1, f"{label_idx=} {fpr.shape=}"
            assert (
                fpr.shape[0] == tpr.shape[0]
            ), f"{label_idx=} {tpr.shape=} {fpr.shape=}"

    labels_idx = list(range(nlabels))

    report_dict = {}

    for label_idx in labels_idx:

        tp = cm[label_idx, label_idx]
        fp = cm[:label_idx, label_idx].sum() + cm[label_idx + 1 :, label_idx].sum()
        fn = cm[label_idx, :label_idx].sum() + cm[label_idx, label_idx + 1 :].sum()
        tn = cm.sum() - tp - fp - fn
        support = cm[label_idx, :].sum()
        npred = cm[:, label_idx].sum()

        assert (sum_ := tp + fp + fn + tn) == cm.sum(), f"{sum_=}"

        report_dict[label_idx] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "tp_relative": tp / cm.sum(),
            "fp_relative": fp / cm.sum(),
            "fn_relative": fn / cm.sum(),
            "tn_relative": tn / cm.sum(),
            "support": support,
            "npred": npred,
            "accuracy": (tp + tn) / cm.sum(),
            "precision": tp / (tp + fp),
            "recall": tp / (tp + fn),
            "f1": 2 * tp / (2 * tp + fp + fn),
            "jaccard": tp / (support + npred - tp),
        }

        if rocs is not None:
            roc = rocs[label_idx]
            fpr, tpr = roc["fpr"], roc["tpr"]
            report_dict[label_idx].update({"roc-auc": metrics.auc(fpr, tpr)})

    macro_avg_metrics = ("accuracy", "precision", "recall", "f1", "jaccard")

    if rocs is not None:
        macro_avg_metrics = tuple(list(macro_avg_metrics) + ["roc-auc"])

    report_dict["macro"] = {
        m: np.average([report_dict[label_idx][m] for label_idx in labels_idx])
        for m in macro_avg_metrics
    }

    report_dict["micro"] = {
        m: np.sum([report_dict[label_idx][m] for label_idx in labels_idx])
        for m in ("tp", "fp", "fn", "tn")
    }

    tp_, fp_, fn_, tn_ = (report_dict["micro"][m] for m in ("tp", "fp", "fn", "tn"))

    report_dict["micro"].update(
        {
            "accuracy": tp_ / cm.sum(),
            "precision": tp_ / (tp_ + fp_),
            "recall": tp_ / (tp_ + fn_),
            "f1": 2 * tp_ / (2 * tp_ + fp_ + fn_),
        }
    )

    # convert to float...
    for key, subdic in report_dict.items():
        for subkey, val in subdic.items():
            report_dict[key][subkey] = (
                int(val) if isinstance(val, np.int64) else float(val)
            )

    return report_dict


class ClassifReportHumandDumper(yaml.Dumper):
    pass


def percentage_float_representer(dumper, value):
    return dumper.represent_scalar("tag:yaml.org,2002:str", f"{value:.2%}")


def humanize_int_representer(dumper, value):
    return dumper.represent_scalar(
        "tag:yaml.org,2002:str",
        f"{humanize.intword(value)} ({humanize.intcomma(value)})",
    )


ClassifReportHumandDumper.add_representer(float, percentage_float_representer)
ClassifReportHumandDumper.add_representer(int, humanize_int_representer)


def multiclass_roc_auc_score(
    y_true,
    y_score,
    labels,
    multi_class,
    average,
    sample_weight=None,
    invalid_proba_tolerance: float = 1e-6,
):
    """Multiclass roc auc score (copied from sklearn)

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True multiclass labels.

    y_score : array-like of shape (n_samples, n_classes)
        Target scores corresponding to probability estimates of a sample
        belonging to a particular class

    labels : array, shape = [n_classes] or None, optional (default=None)
        List of labels to index ``y_score`` used for multiclass. If ``None``,
        the lexical order of ``y_true`` is used to index ``y_score``.

    multi_class : string, 'ovr' or 'ovo'
        Determines the type of multiclass configuration to use.
        ``'ovr'``:
            Calculate metrics for the multiclass case using the one-vs-rest
            approach.
        ``'ovo'``:
            Calculate metrics for the multiclass case using the one-vs-one
            approach.

    average : 'macro' or 'weighted', optional (default='macro')
        Determines the type of averaging performed on the pairwise binary
        metric scores
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean. This does not take label imbalance into account. Classes
            are assumed to be uniformly distributed.
        ``'weighted'``:
            Calculate metrics for each label, taking into account the
            prevalence of the classes.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    :param invalid_proba_tolerance: float in [0, 1]
        The proportion of samples that can eventually be ignored if their class scores do not sum up to 1.
    """
    # validation of the input y_score
    are_close = np.isclose(1, y_score.sum(axis=1))

    # I added this try-except to deal with cases where a very small amount of voxels have an issue
    # to sum the probabilities to 1, which might happen (probably, i suppose) because I use float16 instead of 64
    try:
        if not np.all(are_close):
            raise ValueError(
                "Target scores need to be probabilities for multiclass "
                "roc_auc, i.e. they should sum up to 1.0 over classes"
            )

    except ValueError as ex:

        logger.exception(ex)

        assert 0 <= invalid_proba_tolerance <= 1, f"{invalid_proba_tolerance=}"

        nsamples_not_close = int((~are_close).sum())
        percentage_samples_not_close = nsamples_not_close / are_close.size

        logger.warning(f"{nsamples_not_close=} ({percentage_samples_not_close=:.7%})")

        if percentage_samples_not_close > invalid_proba_tolerance:
            raise ValueError(
                f"Too many samples are not close 1 {nsamples_not_close=} {percentage_samples_not_close=:.7%} {invalid_proba_tolerance=:.7%}."
            )

        else:
            logger.warning(
                f"The amount of probabilities not summing up to 1 will be tolerated "
                f"{percentage_samples_not_close=:.7%} {invalid_proba_tolerance=:.7%}. "
                f"The bad samples will be ignored!"
            )

            y_true = y_true[are_close]
            y_score = y_score[are_close, :]

    # validation for multiclass parameter specifications
    average_options = ("macro", "weighted")
    if average not in average_options:
        raise ValueError(
            "average must be one of {0} for "
            "multiclass problems".format(average_options)
        )

    multiclass_options = ("ovo", "ovr")
    if multi_class not in multiclass_options:
        raise ValueError(
            "multi_class='{0}' is not supported "
            "for multiclass ROC AUC, multi_class must be "
            "in {1}".format(multi_class, multiclass_options)
        )

    from sklearn.utils import column_or_1d
    from sklearn.preprocessing._label import _encode
    from sklearn.metrics._base import _average_multiclass_ovo_score
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics._ranking import _binary_roc_auc_score
    from sklearn.metrics._base import _average_binary_score

    if labels is not None:
        labels = column_or_1d(labels)
        classes = _encode(labels)
        if len(classes) != len(labels):
            raise ValueError("Parameter 'labels' must be unique")
        if not np.array_equal(classes, labels):
            raise ValueError("Parameter 'labels' must be ordered")
        if len(classes) != y_score.shape[1]:
            raise ValueError(
                "Number of given labels, {0}, not equal to the number "
                "of columns in 'y_score', {1}".format(len(classes), y_score.shape[1])
            )
        if len(np.setdiff1d(y_true, classes)):
            raise ValueError("'y_true' contains labels not in parameter 'labels'")
    else:
        classes = _encode(y_true)
        if len(classes) != y_score.shape[1]:
            raise ValueError(
                "Number of classes in y_true not equal to the number of "
                "columns in 'y_score'"
            )

    if multi_class == "ovo":
        if sample_weight is not None:
            raise ValueError(
                "sample_weight is not supported "
                "for multiclass one-vs-one ROC AUC, "
                "'sample_weight' must be None in this case."
            )
        _, y_true_encoded = _encode(y_true, uniques=classes, encode=True)
        # Hand & Till (2001) implementation (ovo)
        return _average_multiclass_ovo_score(
            _binary_roc_auc_score, y_true_encoded, y_score, average=average
        )
    else:
        # ovr is same as multi-label
        y_true_multilabel = label_binarize(y_true, classes=classes)
        return _average_binary_score(
            _binary_roc_auc_score,
            y_true_multilabel,
            y_score,
            average,
            sample_weight=sample_weight,
        )


def _get_slice_props_parallel_do(
    slice_idx_slice_tup: Tuple[int, slice],
    label_volume: ndarray,
    data_volume: ndarray,
    normal_axis: int,
) -> Dict[str, ndarray]:

    slice_idx, slice_ = slice_idx_slice_tup

    props = skimage.measure.regionprops_table(
        label_image=skimage.measure.label(
            label_volume[slice_], connectivity=1, background=0, return_num=False,
        ),
        intensity_image=data_volume[slice_],
        properties=(
            "area",
            "bbox",
            "centroid",
            "major_axis_length",
            "minor_axis_length",
        ),
        cache=True,
        separator="-",
    )

    props["normal_axis"] = np.full_like(props["area"], normal_axis, dtype=np.int8)
    props["slice_idx"] = np.full_like(props["area"], slice_idx)

    return props


def get_slice_props_parallel(
    label_volume: ndarray,
    data_volume: ndarray,
    normal_axis: int,
    nprocs: Optional[int] = None,
    chunksize: int = 10,
) -> Dict[str, List[Union[int, float]]]:

    import sys

    assert label_volume.shape == data_volume.shape
    assert 0 <= normal_axis <= 2

    nslices = label_volume.shape[normal_axis]

    logger.debug(f"{normal_axis=} => {nslices=}")

    slices = []

    for slice_idx in range(nslices):

        slice_ = 3 * [slice(None)]
        slice_[normal_axis] = slice(slice_idx, slice_idx + 1)
        slice_ = tuple(slice_)

        slices.append(slice_)

    func = partial(
        _get_slice_props_parallel_do,
        label_volume=label_volume,
        data_volume=data_volume,
        normal_axis=normal_axis,
    )

    logger.debug(f"processing slices {nprocs=}")

    blobs2d_props = []

    with Pool(nprocs) as p:

        for i, props in enumerate(
            p.imap_unordered(func, enumerate(slices), chunksize=chunksize), 1
        ):
            sys.stderr.write(f"\rdone {i / nslices:.0%}")
            blobs2d_props.append(props)

    logger.debug("done => merging all slices' props dicts")

    return {
        k: np.concatenate([prop_dic[k] for prop_dic in blobs2d_props]).tolist()
        for k in blobs2d_props[0].keys()
    }


def add_notable_slices(
    func: Callable[[pd.DataFrame], Tuple[dict, dict]],
    notable_slices: dict,
    error_2dblobs_props: DataFrame,
    axes=(0, 1, 2),
):

    for axis in axes:

        name, slice_idx, custom_attrs = func(error_2dblobs_props.loc[axis])

        name += f".{axis=}"

        notable_slice_dict = notable_slices[name] = {
            "name": name,
            "normal_axis": axis,
            "slice_idx": int(slice_idx),
        }

        slice_ = [slice(None), slice(None), slice(None)]
        slice_[axis] = slice(slice_idx, slice_idx + 1)
        slice_ = tuple(slice_)

        notable_slice_dict.update(
            {"slice": slice_, **custom_attrs,}
        )


def add_notable_slices_blobwise(
    func: Callable[[pd.DataFrame], Tuple[dict, dict]],
    notable_slices: dict,
    error_2dblobs_props: DataFrame,
    axes=(0, 1, 2),
):
    for axis in axes:

        name, row, custom_attrs = func(error_2dblobs_props.loc[axis])

        name += f".{axis=}"

        notable_slice_dict = notable_slices[name] = {
            "name": name,
            "normal_axis": axis,
            "slice_idx": int(row.slice_idx),
        }

        centroid3d = tuple(int(row[f"centroid-{axx}"]) for axx in range(3))

        centroid2d = tuple(val for axx, val in enumerate(centroid3d) if axx != axis)

        bbox3d = (
            slice(int(row[f"bbox-0"]), int(row[f"bbox-3"])),
            slice(int(row[f"bbox-1"]), int(row[f"bbox-4"])),
            slice(int(row[f"bbox-2"]), int(row[f"bbox-5"])),
        )

        bbox2d = tuple(val for axx, val in enumerate(bbox3d) if axx != axis)

        notable_slice_dict.update(
            {
                "centroid3d": centroid3d,
                "centroid2d": centroid2d,
                "bbox3d": bbox3d,
                "bbox2d": bbox2d,
                **custom_attrs,
            }
        )


def max_area(df):
    row = df.iloc[df.area.argmax()]
    custom_attrs = {"blob-area": int(row["area"])}
    return "error-blob.max-area", row, custom_attrs


def max_bbox_shape(df, dim):
    """bbox = bounding box"""

    bbox_shape_dim = df[f"bbox-{dim + 3}"] - df[f"bbox-{dim}"]

    arg_max = bbox_shape_dim.argmax()
    row = df.iloc[arg_max]

    custom_attrs = {f"blob-bbox.length.axis={dim}": int(bbox_shape_dim.iloc[arg_max])}

    return f"error-blob.max-bbox-lenghth-axis={dim}", row, custom_attrs


def max_major_axis_length(df):
    row = df.iloc[df.major_axis_length.argmax()]
    custom_attrs = {"blob-major-axis-length": float(row["major_axis_length"])}
    return "error-blob.max-major-axis-length", row, custom_attrs


def max_minor_axis_length(df):
    row = df.iloc[df.minor_axis_length.argmax()]
    custom_attrs = {"blob-minor-axis-length": float(row["minor_axis_length"])}
    return "error-blob.max-minor-axis-length", row, custom_attrs


def max_error_area(df):
    area_per_slice = df[["area", "slice_idx"]].groupby("slice_idx").sum()
    slice_idx = area_per_slice.area.argmax()
    custom_attrs = {"slice-error-area": int(area_per_slice.iloc[slice_idx].area)}
    return "max-error-area", slice_idx, custom_attrs


def max_error_blob_avg_area(df):
    avg_blob_area_per_slice = df[["area", "slice_idx"]].groupby("slice_idx").mean()
    slice_idx = avg_blob_area_per_slice.area.argmax()
    custom_attrs = {
        "slice-avg-error-blob-area": int(avg_blob_area_per_slice.iloc[slice_idx].area)
    }
    return "max-avg-error-blob-area", slice_idx, custom_attrs


def get_2d_blob_props(
    label_volume: ndarray,
    data_volume: ndarray,
    axes: Tuple[int] = (0, 1, 2),
    parallel_nprocs: Optional[int] = None,
) -> DataFrame:
    assert min(axes) >= 0, f"{min(axes)=}"
    assert max(axes) <= 2, f"{max(axes)=}"

    all_blob_props = []

    for axis in axes:
        logger.info(f"computing 2d_blob_props on plane normal to {axis=}")
        all_blob_props.append(
            get_slice_props_parallel(
                label_volume, data_volume, normal_axis=axis, nprocs=parallel_nprocs,
            )
        )

    logger.debug("Converting 2d blob props dicts to data frames.")

    for axis in axes:

        blob_props = all_blob_props[axis]

        ref_shape = len(blob_props["area"])

        for k in blob_props.keys():
            assert (
                shap := len(blob_props[k])
            ) == ref_shape, f"{k=} {shap=} {ref_shape=}"

        all_blob_props[axis] = pd.DataFrame(blob_props)

        logger.debug(f"{all_blob_props[axis].shape=}")

    return pd.concat(all_blob_props, axis=0)
