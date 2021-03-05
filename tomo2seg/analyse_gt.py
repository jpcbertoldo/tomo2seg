import time
from abc import ABC
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, ClassVar, Type, Tuple, List, Callable
from numpy import ndarray
import numpy as np

import tensorflow as tf

from tomo2seg.logger import logger, dict2str
from tomo2seg.hosts import Host, get_host
from tomo2seg.args import BaseArgs
from tomo2seg.outputs import BaseOutputs, mkdir_ok
from pathlib import Path


MAX_BIN_EDGE = {
    "uint8": 256,
    "uint16": 65536,
}


@dataclass
class AnalyseGroundTruthOuputs(BaseOutputs):
    
    @property
    def histogram_per_label_bins(self) -> Path:
        return "histogram-per-label.bins.npy"
        
    def histogram_per_label(self, partition_: Optional[str]) -> Path:
        fname = f"histogram-per-label"
        fname += f".partition={partition_}" if partition_ is not None else ""
        fname += ".npy"
        return self.root_dir / fname
    
    @property
    def class_imbalance_plot(self) -> Path:
        return self.root_dir / "class-imbalance.png"
    
    def histogram_plot(self, partition_: Optional[str]) -> Path:
        fname = f"histogram"
        fname += f".partition={partition_}" if partition_ is not None else ""
        fname += ".png"
        return self.root_dir / fname
        
    def histogram_per_label_plot(self, partition_: Optional[str]) -> Path:
        fname = f"histogram-per-label"
        fname += f".partition={partition_}" if partition_ is not None else ""
        fname += ".png"
        return self.root_dir / fname
    
    @property
    @mkdir_ok
    def layers_correlation_dir(self) -> Path:
        return self.root_dir / "layers-correlation"
        
    def layers_correlation(self, axis: int, nlayers: int, label: Optional[int]) -> Path:
        label = "all" if label is None else label
        return self.layers_correlation_dir / f"layers-correlation.label={label}.{axis=}.{nlayers=}.npy"
    
    @property
    def layers_correlation_plot(self) -> Path:
        return self.root_dir / f"layers-correlation.png"
    
    @property
    @mkdir_ok
    def layerwise_class_count_dir(self) -> Path:
        return self.root_dir / "layerwise-class-count"
    
    def layerwise_class_count(self, axis: int) -> Path:
        return self.layerwise_class_count_dir / f"layerwise-class-count.{axis=}.npy"
    
    @property
    def layerwise_class_count_plot(self) -> Path:
        return self.root_dir / f"layerwise-class-count.png"


@dataclass
class AnalyseGroundTruthMetaArgs(BaseArgs):
    
    volume_name: str
    volume_version: str
    labels_version: str
    partitions_to_compute: Optional[Tuple[str]] = None   
        
    def __post_init__(self):
        
        # todo: move me to the parent class
        logger.info(f"{self.__class__.__name__}\n{dict2str(asdict(self))}")
        
        
def validate_partitions_to_compute(partitions_to_compute, volume):
    
    if partitions_to_compute is None:

        logger.info("Using all available parittions.")

        return tuple(volume.metadata.set_partitions.keys())

    assert len(partitions_to_compute) > 0

    for part_alias in partitions_to_compute:

        try:
            volume[part_alias]

        except KeyError as ex:
            logger.exception(ex)
            raise ValueError(
                f"Invalid volume partition. {volume.fullname=} {partitions_to_compute=}"
            )
    
    return tuple(partitions_to_compute)
        
        
# todo: move me to data
def partition2slice(partition):
    
    return (
        slice(partition.x_range[0], partition.x_range[1], None),
        slice(partition.y_range[0], partition.y_range[1], None),
        slice(partition.z_range[0], partition.z_range[1], None),
    )


def get_hist_per_label(
    data_seq: ndarray, 
    labels_seq: ndarray,
    labels_idx: List[int],
    nbins: int = 256,
    min_bin_edge: int = 0, 
    max_bin_edge: Optional[int] = None,  # auto from data dtype
):
    """
    data_seq: gray level data in a sequential vector
    labels_seq: segmentation classes in a sequential vector
    """
    logger.debug("computing histogram per label")
    
    assert (tensor_order := len(data_seq.shape)) == 1, f"{tensor_order=}"
    assert (tensor_order := len(labels_seq.shape)) == 1, f"{tensor_order=}"
    
    logger.debug(f"{data_seq.shape=}")
    logger.debug(f"{labels_seq.shape=}")
    
    assert len(labels_idx) > 0, f"{len(labels_idx)=}"
    assert all(isinstance(v, int) for v in labels_idx), f"{labels_idx=}"
    
    logger.debug(f"{labels_idx=}")
    
    nclasses = len(labels_idx)
   
    logger.debug(f"{nclasses=}")
    
    assert nbins > 1, f"{nbins=}"
    
    logger.debug(f"{nbins=}")
    
    assert min_bin_edge >= 0, f"{min_bin_edge=}"
    
    dtype = str(data_seq.dtype)
    auto_max_bin_edge = MAX_BIN_EDGE[dtype]
    
    if max_bin_edge is None:
        max_bin_edge = auto_max_bin_edge
    
    else:
        assert max_bin_edge <= auto_max_bin_edge, f"{max_bin_edge=}, give {dtype=} ==> max is {auto_max_bin_edge=}"

    logger.debug(f"{min_bin_edge=}")
    logger.debug(f"{max_bin_edge=}")
    
    # --------------------------- real stuff ---------------------------

    data_hists_per_label = np.zeros(
        (nclasses, nbins), 
        dtype=np.int64,  # int64 is important to not overflow
    ) 
    
    hist_bin_edges = np.linspace(min_bin_edge, max_bin_edge, nbins + 1).astype(int)

    for label_idx in labels_idx:

        logger.debug(f"computing {label_idx=}")

        data_hists_per_label[label_idx], _ = np.histogram(
            data_seq[labels_seq == label_idx],
            bins=hist_bin_edges,
            density=False,
        )
        
    return data_hists_per_label, hist_bin_edges


def jaccard(u: ndarray, v: ndarray, label: Optional[int] = None) -> float:
    """
    label: class to consider, if none then all confounded
    """
    assert u.shape == v.shape, f"{u.shape=} {v.shape=}"
    assert np.issubdtype(u.dtype, np.integer), f"{u.dtype=}"
    assert u.dtype == v.dtype, f"{u.dtype=} {v.dtype=}"
    
    if label is not None:
        
        assert isinstance(label, int), f"{u.dtype=} {type(label)=}"
        
        u = (u == label)
        v = (v == label)
    
    intersec = (u & v).sum() if label is not None else (u == v).sum()
    
    union = (u | v).sum() if label is not None else u.size
    
    return intersec / union


def adjacent_layers_correlation(
    labels: ndarray,
    axis: int,
    nslices: int,
    correlation_func: Callable[[ndarray, ndarray], float] 
):

    assert labels.ndim == 3, f"{labels.ndim=}"
    
    assert 0 <= axis <= 2, f"{axis=}"
    logger.debug(f"{axis=}")
    
    axis_size = labels.shape[axis]
    logger.debug(f"{axis_size=}")
    
    assert 1 <= nslices <= axis_size - 1, f"{nslices}"
    logger.debug(f"{nslices=}")
    
    logger.debug(f"{correlation_func=}")

    def get_slice(idx: int) -> slice:
        slice_ = 3 * [slice(None, None, None)]
        slice_[axis] = slice(idx, idx + nslices, None)
        return tuple(slice_)

    corrs = [
        correlation_func(
            labels[get_slice(idx)],
            labels[get_slice(idx + 1)],
        )
        for idx in range(axis_size - nslices)
    ]
    
    assert all(0 <= val <= 1 for val in corrs), f"Issue with {correlation_func=}"
    
    return corrs


def class_counts_per_layer(labels: ndarray, axis: int, nclasses: int) -> ndarray:

    assert labels.ndim == 3, f"{labels.ndim=}"

    assert 0 <= axis <= 2, f"{axis=}"
    logger.debug(f"{axis=}")

    axis_size = labels.shape[axis]
    logger.debug(f"{axis_size=}")

    counts = np.empty((axis_size, nclasses), dtype=np.int64)
    logger.debug(f"{counts.shape=}")

    for label_idx in range(3):

        axes_sum = [0, 1, 2]
        axes_sum.pop(axis)

        counts[:, label_idx] = np.sum((labels == label_idx), axis=tuple(axes_sum))
    
    return counts