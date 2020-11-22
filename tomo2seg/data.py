import time
from datetime import datetime
from dataclasses import dataclass, field
import pathlib
from pathlib import Path
from typing import List, Tuple, Dict, Optional, ClassVar

import yaml
from numpy import ndarray
from yaml import YAMLObject

from .logger import logger

here = pathlib.Path(__file__).parent.resolve().absolute()
root_dir = (here / "..").resolve()
data_dir = (root_dir / "data").resolve()
models_dir = (data_dir / "models").resolve()

data_dir.mkdir(exist_ok=True)
models_dir.mkdir(exist_ok=True)


@dataclass
class SetPartition(YAMLObject):
    x_range: Tuple[int, int]
    y_range: Tuple[int, int]
    z_range: Tuple[int, int]
    alias: Optional[str]

    yaml_tag = "!SetPartition"

    @property
    def canonical_alias(self) -> str:
        return f"x:{self.x_range[0]}:{self.x_range[1]}-" \
               f"y:{self.y_range[0]}:{self.y_range[1]}-" \
               f"z:{self.z_range[0]}:{self.z_range[1]}"

    @classmethod
    def z_partitioned_only(cls, x_size: int, y_size: int, z_min: int, z_max: int, alias=None) -> "SetPartition":
        return cls((0, x_size), (0, y_size), (z_min, z_max), alias=alias)

    @classmethod
    def from_dict(cls, dic: dict):
        for k, v in dic.items():
            if "range" in k:
                assert len(v) == 2
                dic[k] = tuple(v)
        return cls(**dic)

    def get_volume_partition(self, volume: ndarray) -> ndarray:
        return volume[
            self.x_range[0]:self.x_range[1],
            self.y_range[0]:self.y_range[1],
            self.z_range[0]:self.z_range[1],
        ]

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (
            self.x_range[1] - self.x_range[0],
            self.y_range[1] - self.y_range[0],
            self.z_range[1] - self.z_range[0],
        )

    @property
    def n_voxels(self) -> int:
        return self.shape[0] * self.shape[1] * self.shape[2]


@dataclass
class Volume:
    """
    example of usage

    # prefill the function `HST_read`
    _hst_read = functools.partial(
        file_utils.HST_read,
        autoparse_filename=False,  # the file names are not properly formatted
        data_type=dtype,
        dims=dimensions,
        verbose=True,
    )

    # adapt it to get paths
    hst_read = lambda x: _hst_read(str(x))

    volume = Volume.with_check(*tomo2seg_data.VOLUME_PRECIPITATES_V1)
    data_volume = hst_read(volume.data_path)
    labels_volume = hst_read(volume.labels_path)

    train_data = volume.train_partition.get_volume_partition(data_volume)
    train_labels = volume.train_partition.get_volume_partition(labels_volume)

    val_data = volume.val_partition.get_volume_partition(data_volume)
    val_labels = volume.val_partition.get_volume_partition(labels_volume)
    """

    def xy_reduced(self, new_width: int, new_height: int, alias=None) -> SetPartition:
        alias = alias if alias is not None else f"xy_reduced({new_width}, {new_height})"
        return SetPartition(
            (0, new_width), (0, new_height), (0, self._metadata.dimensions[2]),
            alias=alias
        )

    @dataclass
    class Metadata(YAMLObject):
        yaml_tag = "!Volume.Metadata"

        TRAIN_PARTITION_KEY = "train"
        VAL_PARTITION_KEY = "val"
        TEST_PARTITION_KEY = "test"

        dimensions: Tuple[int, int, int]
        dtype: str
        labels: List[int]
        labels_names: Dict[int, str]
        set_partitions: Optional[Dict[str, SetPartition]] = None

    name: str
    version: Optional[str] = None
    _metadata: Optional["Volume.Metadata"] = None

    @property
    def fullname(self) -> str:
        if self.version is not None:
            return f"{self.name}.{self.version}"
        else:
            return f"{self.name}"

    @property
    def dir(self) -> Path:
        return data_dir / f"{self.fullname}"

    @property
    def metadata_path(self) -> Path:
        return self.dir / f"{self.fullname}.metadata.yml"

    @property
    def data_path(self) -> Path:
        return self.dir / f"{self.fullname}.raw"

    @property
    def info_path(self) -> Path:
        return self.dir / f"{self.fullname}.raw.info"

    @property
    def labels_path(self) -> Path:
        return self.dir / f"{self.fullname}.labels.raw"

    def versioned_labels_path(self, version_suffix: Optional[str] = None) -> Path:
        if version_suffix is None:
            return self.labels_path
        return self.dir / f"{self.fullname}.labels-{version_suffix}.raw"

    def _blobs_path_prefix(self, labels_version: Optional[str]) -> str:
        if labels_version is not None:
            labels_volume_path = self.versioned_labels_path(labels_version)
        else:
            labels_volume_path = self.labels_path
        return str(labels_volume_path)[:-4]

    def blobs3d_volume_path(self, class_idx: int, labels_version: Optional[str] = None) -> Path:
        return self.dir / f"{self._blobs_path_prefix(labels_version)}.blobs3d.class_idx={class_idx}.raw"

    def blobs3d_props_path(self, class_idx: int, labels_version: Optional[str] = None) -> Path:
        return self.dir / f"{self._blobs_path_prefix(labels_version)}.blobs3d.props.class_idx={class_idx}.csv"

    @property
    def weights_path(self) -> Path:
        return self.dir / f"{self.fullname}.weights.raw"

    def versioned_weights_path(self, version_suffix: Optional[str] = None) -> Path:
        if version_suffix is None:
            return self.weights_path
        return self.dir / f"{self.fullname}.weights-{version_suffix}.raw"

    def volume_processing_dir(self, execid: str) -> Path:
        return self.dir / f"process-volume.execution={execid}"

    @property
    def metadata(self) -> "Volume.Metadata":

        if self._metadata is None:
            logger.debug(f"Loading metadata from `{self.metadata_path}`.")
            with self.metadata_path.open("r") as file:
                self._metadata = yaml.load(file, Loader=yaml.FullLoader)

        return self._metadata

    def write_metadata(self, key, value):
        logger.debug(f"Writing to metadata file at `{self.metadata_path}`")

        if self._metadata is None:
            self.metadata  # load it

        setattr(self._metadata, key, value)

        with self.metadata_path.open("w") as f:
            yaml.dump(self._metadata, f, default_flow_style=False, indent=4)

    @property
    def set_partitions(self) -> Dict[str, SetPartition]:
        return self.metadata.set_partitions

    @property
    def train_partition(self) -> SetPartition:
        return SetPartition.from_dict(
            self.set_partitions[self.Metadata.TRAIN_PARTITION_KEY]
        )

    @property
    def val_partition(self) -> SetPartition:
        return SetPartition.from_dict(
            self.set_partitions[self.Metadata.VAL_PARTITION_KEY]
        )

    @property
    def test_partition(self) -> SetPartition:
        return SetPartition.from_dict(
            self.set_partitions[self.Metadata.TEST_PARTITION_KEY]
        )

    @classmethod
    def with_check(cls, name, version: str = None):
        vol: "Volume" = cls(name=name, version=version)

        logger.debug(f"{vol=}")

        # the minimal files required
        error_paths: List[Path] = [
            vol.data_path,
            vol.labels_path,
            vol.info_path,
            vol.metadata_path,
        ]

        # these are not essential but important
        warning_paths: List[Path] = [
            # train
            vol.weights_path,
        ]

        for p in error_paths:
            if not p.exists():
                logger.error("Missing file: %s", str(p))

        for p in warning_paths:
            if not p.exists():
                logger.warning("Missing file: %s", str(p))

        if vol.set_partitions is not None:

            for key in [
                cls.Metadata.TRAIN_PARTITION_KEY, cls.Metadata.VAL_PARTITION_KEY, cls.Metadata.TEST_PARTITION_KEY
            ]:
                if key not in vol.set_partitions:
                    logger.warning(f"Missing set partition: {key=}")

        return vol


@dataclass
class EstimationVolume(YAMLObject):
    """"""

    yaml_tag: ClassVar[str] = "tomo2seg.EstimationVolume"

    volume_fullname: str
    model_name: str
    runid: int = field(default_factory=lambda: int(time.time()))
    partition: Optional[SetPartition] = None

    @property
    def fullname(self) -> str:
        partition_name = "whole-volume" if self.partition is None else (
            self.partition.alias or self.partition.canonical_alias
        )
        s = str(self.runid)
        return f"vol={self.volume_fullname}.set={partition_name}.model={self.model_name}.runid={s[:4]}-{s[4:7]}-{s[7:]}"

    @property
    def dir(self) -> Path:
        (dir_ := data_dir / self.fullname).mkdir(exist_ok=True)
        return dir_

    @property
    def metadata_path(self) -> Path:
        pth = self.dir / f"{self.fullname}.metadata.yml"
        if not pth.exists():
            logger.debug(f"Creating metadata file {pth}.")
            pth.touch()
        return pth

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        logger.debug(f"Writing to file {self.metadata_path=}.")
        setattr(self, key, value)
        with self.metadata_path.open("w") as f:
            yaml.dump(self, f, default_flow_style=False, indent=4)

    @property
    def debug__crops_coordinates_path(self) -> Path:
        return self.dir / f'{self.fullname}.debug.crops-coordinates.npy'

    @property
    def debug__crops_path(self) -> Path:
        return self.dir / f"{self.fullname}.debug.crops.npy"

    @property
    def debug__crops_probas_path(self) -> Path:
        return self.dir / f"{self.fullname}.debug.crops-probabilities.npy"

    @property
    def debug__crops_preds_path(self) -> Path:
        return self.dir / f"{self.fullname}.debug.crops-predictions.npy"

    @property
    def probabilities_path(self) -> Path:
        return self.dir / f"{self.fullname}.probabilities.npy"

    def get_class_probability_path(self, class_idx: int) -> Path:
        return self.dir / f"{self.fullname}.probability.class-idx={class_idx}.raw"

    @property
    def predictions_path(self) -> Path:
        return self.dir / f"{self.fullname}.predictions.raw"

    @property
    def presoftmax_voxel_embeddings_path(self) -> Path:
        return self.dir / f"{self.fullname}.presoftmax-voxel-embeddings.npy"
    
    @property
    def voxelwise_classification_report_human(self) -> Path:
        return self.dir / f"{self.fullname}.voxelwise-classification-report.human.yaml"

    @property
    def voxelwise_classification_report_exact(self) -> Path:
        return self.dir / f"{self.fullname}.voxelwise-classification-report.exact.yaml"
    
    def get_class_roc_curve_path(self, class_idx: int) -> Path:
        return self.dir / f"{self.fullname}.roc-curve.class-idx={class_idx}.raw"

    @property
    def binary_confusion_matrices_path(self) -> Path:
        return self.dir / f"{self.fullname}.binary-confusion-matrices.npy"

    @property
    def confusion_matrix_path(self) -> Path:
        return self.dir / f"{self.fullname}.confusion-matrix.npy"

    @property
    def probabilities_histograms_path(self) -> Path:
        return self.dir / f"{self.fullname}.probabilities-histograms.npy"
    
    @property
    def voxel_normalized_entropy_path(self) -> Path:
        return self.dir / f"{self.fullname}.voxel-normalized-entropy.raw"
    
    @property
    def voxel_normalized_entropy_histograms_path(self) -> Path:
        return self.dir / f"{self.fullname}.voxel-normalized-entropy-histograms.npy"
    
    def get_confusion_volume_path(self, class_idx) -> Path:
        return self.dir / f"{self.fullname}.confusion-volume.class-idx={class_idx}.raw"
    
    @classmethod
    def from_objects(cls, volume: Volume, model: "Model", set_partition: SetPartition = None, runid=None):
        return cls(
            volume_fullname=volume.fullname,
            model_name=model.name,
            partition=set_partition,
            runid=runid
        )
