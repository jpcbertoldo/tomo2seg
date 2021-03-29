import itertools
import time
from abc import ABC
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, ClassVar, Dict, List, Optional, Tuple, Type

import humanize
import numpy as np
import pandas as pd
import tensorflow as tf
from numpy import ndarray
from pandas import DataFrame
from tomo2seg.args import BaseArgs
from tomo2seg.hosts import Host, get_host
from tomo2seg.logger import dict2str, logger
from tomo2seg.outputs import BaseOutputs, mkdir_ok


@dataclass
class AnalysePredOpts:
    """this can be used both by the script and the meta args"""

    @dataclass
    class Compute:
        """things that are optionally computed"""

        error_volume: bool = True

        roc_curve: bool = True
        multiclass_roc_auc: bool = True

        error_blobs_2d_props: bool = True
        error_blobs_3d_props: bool = True

        adjacent_layers_correlation: bool = True

        def assert_dependency(self, master: str, dependent: str):

            master_value = getattr(self, master)
            dependent_value = getattr(self, dependent)

            if dependent_value == True:
                assert (
                    master_value == True
                ), f"{dependent=}={dependent_value} but {master=}={master_value}"

        def __post_init__(self):

            # dependencies
            self.assert_dependency("error_volume", "error_blobs_2d_props")

            if self.error_blobs_3d_props == True:
                raise NotImplementedError(f"{self.error_blobs_3d_props=}")

    @dataclass
    class Save:
        """things that are optionally saved"""

        confusion_volume: bool = True
        error_volume: bool = True

        error_blobs_2d_props: bool = True
        error_blobs_3d_props: bool = True

    # --- opts ---
    compute: Compute = field(default_factory=Compute)
    save: Save = field(default_factory=Save)

    def __post_init__(self):

        saveables = list(self.Save.__dataclass_fields__.keys())
        computables = list(self.Compute.__dataclass_fields__.keys())

        needs_to_be_computed_if_saved = sorted(set(saveables) & set(computables))

        for field in needs_to_be_computed_if_saved:
            if getattr(self.save, field) == True:
                assert (
                    getattr(self.compute, field) == True
                ), f"{field=} must be computed to be saved: {getattr(self.save, field)=} but {getattr(self.compute, field)=}"


@dataclass
class AnalysePredMetaArgs(BaseArgs):

    # versions of the script compatible with this class
    versions: ClassVar = (1, 2)

    volume_name: str
    volume_version: str
    labels_version: str

    estimation_volume_fullname: str

    opts: AnalysePredOpts = field(default_factory=AnalysePredOpts)

    def __post_init__(self):
        super().__post_init__()

        # todo: move me to the parent class
        logger.info(f"{self.__class__.__name__}\n{dict2str(asdict(self))}")


@dataclass
class AnalysePredOuputs(BaseOutputs):
    @property
    def confusion_volume_path(self) -> Path:
        return self.root_dir / f"confusion-volume.raw"

    @property
    def error_volume_path(self) -> Path:
        return self.root_dir / f"error-volume.raw"

    @property
    def confusion_matrix(self) -> Path:
        return self.root_dir / f"confusion-matrix.npy"

    def roc_curve(self, class_idx: int) -> Path:
        return self.root_dir / f"roc-curve.class_idx={class_idx}.csv"

    @property
    def error_2dblobs_props(self) -> Path:
        return self.root_dir / f"error-2dblobs-props.csv"

    @property
    def classification_report_human(self) -> Path:
        return self.root_dir / f"classification-report.human.yaml"

    @property
    def classification_report_exact(self) -> Path:
        return self.root_dir / f"classification-report.exact.yaml"

    @property
    def classification_report_table_csv(self) -> Path:
        return self.root_dir / f"classification-report.table.csv"

    @property
    def classification_report_table_human(self) -> Path:
        return self.root_dir / f"classification-report.table.human.txt"

    @property
    def classification_report_table_exact(self) -> Path:
        return self.root_dir / f"classification-report.table.exact.txt"

    @property
    def confusion_matrices_plot(self) -> Path:
        return self.root_dir / f"confusion-matrices.png"

    @property
    def roc_plot(self) -> Path:
        return self.root_dir / f"roc-curves.png"

    @property
    def volumetric_fraction_plot(self) -> Path:
        return self.root_dir / f"volumetric-fraction.png"

    @property
    @mkdir_ok
    def layers_correlation_dir(self) -> Path:
        return self.root_dir / "layers-correlation"

    def layers_correlation(self, axis: int, label: Optional[int]) -> Path:
        label = "all" if label is None else label
        return (
            self.layers_correlation_dir
            / f"layers-correlation.label={label}.{axis=}.npy"
        )

    @property
    def layers_correlation_plot(self) -> Path:
        return self.root_dir / f"layers-correlation.png"

    @property
    @mkdir_ok
    def notable_slices_dir(self) -> Path:
        return self.root_dir / "notable-slices"

    @property
    def notable_slices_yaml(self) -> Path:
        return self.notable_slices_dir / "notable-slices.yaml"

    def notable_slices_plot(self, name: str) -> Path:
        return self.notable_slices_dir / f"{name}.png"


def get_conf_vol_encoding(labels_idx: List[int]) -> Dict[Tuple[int, int], int]:

    nclasses = len(labels_idx)

    assert nclasses > 0, f"{nclasses=} {labels_idx=}"

    if nclasses > 100:
        raise NotImplementedError(f"{nclasses=}")

    assert max(labels_idx) < 100, f"{max(labels_idx)=}"

    # cv = confusion volume
    enc = {
        # (gt, pred)
        (gt_idx, pred_idx): 100 * gt_idx + pred_idx
        for gt_idx, pred_idx in itertools.product(labels_idx, labels_idx)
    }

    inv = dict(map(lambda x: tuple(reversed(x)), enc.items()))

    return enc, inv


def report2table(
    report_dict: dict, labels_names: List[str]
) -> Tuple[DataFrame, DataFrame, DataFrame]:

    table = []
    table_human_simple = []
    table_human_detail = []

    cols0 = ["class/average"]

    cols1 = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc-auc",
        "jaccard",
    ]

    cols2 = [
        "tp",
        "fp",
        "fn",
        "tn",
    ]

    cols3 = [
        "support",
        "npred",
    ]

    cols = cols0 + cols1 + cols2 + cols3

    for key in labels_names + ["macro", "micro"]:

        dic = report_dict[key]

        line = [key] + [dic.get(m, None) for m in cols1 + cols2 + cols3]

        line_human_simple = [
            v
            if isinstance(v, str)
            else (
                f"{humanize.intword(v)}"
                + (
                    f" ({dic[col_rel]:.1%})"
                    if col in ("tp", "fp", "fn", "tn")
                    and (col_rel := col + "_relative") in dic
                    else ""
                )
            )
            if isinstance(v, int)
            else f"{v:.1%}"
            if v is not None
            else "-"
            for v, col in zip(line, cols)
        ]

        line_human_detail = [
            v
            if isinstance(v, str)
            else (
                f"{humanize.intcomma(v)} ({humanize.intword(v)})"
                + (
                    f" ({dic[col_rel]:.4%})"
                    if col in ("tp", "fp", "fn", "tn")
                    and (col_rel := col + "_relative") in dic
                    else ""
                )
            )
            if isinstance(v, int)
            else f"{v:.4%}"
            if v is not None
            else "-"
            for v, col in zip(line, cols)
        ]

        table.append(line)
        table_human_simple.append(line_human_simple)
        table_human_detail.append(line_human_detail)

    table_human_simple.insert(-2, [])
    table_human_detail.insert(-2, [])

    return (
        pd.DataFrame(table, columns=cols).set_index(cols0[0]),
        table_human_simple,
        table_human_detail,
    )
