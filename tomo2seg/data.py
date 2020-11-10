from collections import namedtuple
from datetime import datetime
from dataclasses import dataclass
import pathlib
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import yaml
from numpy import ndarray
from yaml import YAMLObject

from .logger import logger

here = pathlib.Path(__file__).parent.resolve().absolute()
root_dir = (here / "..").resolve()
data_dir = (root_dir / "data").resolve()
models_dir = (root_dir / "models").resolve()

data_dir.mkdir(exist_ok=True)
models_dir.mkdir(exist_ok=True)


@dataclass
class SetPartition(YAMLObject):
    x_range: Tuple[int, int]
    y_range: Tuple[int, int]
    z_range: Tuple[int, int]
    alias: Optional[str]

    yaml_tag = "!SetPartition"

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

    def versioned_labels_path(self, version_suffix) -> Path:
        return self.dir / f"{self.fullname}.labels-{version_suffix}.raw"

    @property
    def weights_path(self) -> Path:
        return self.dir / f"{self.fullname}.weights.raw"

    def versioned_weights_path(self, version_suffix) -> Path:
        return self.dir / f"{self.fullname}.weights-{version_suffix}.raw"

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


# VOLUME NAMES / VERSIONS

#    precipitates dryrun
VOLUME_PRECIPITATES_DRYRUN = "PA66GF30_trans3_x__0_pag"

#    precipitates
VOLUME_PRECIPITATES_V1 = "PA66GF30", "v1"


@dataclass
class ModelPaths:
    
    name: str
    version: str = None
        
    @property
    def fullname(self) -> str:
        if self.version is not None:
            return f"{self.name}.{self.version}"
        else:
            return self.name

    @property
    def model_path(self) -> Path:
        return models_dir / f"{self.fullname}"
    
    @property
    def model_path_str(self) -> str:
        return str(self.model_path)

    @property
    def autosaved_model_path(self) -> Path:
        return models_dir / f"{self.fullname}-autosaved"
    
    @property
    def autosaved_model_path_str(self) -> str:
        return str(self.autosaved_model_path)

    @property
    def logger_path(self) -> Path:
        return self.model_path / "logger.csv"

    @property
    def history_path(self) -> Path:
        return self.model_path / "history.csv"

    @property
    def summary_path(self) -> Path:
        return self.model_path / "summary.txt"

    @property
    def architecture_plot_path(self) -> Path:
        return self.model_path / "architecture.png"

    @property
    def metadata_yml_path(self) -> Path:
        return self.model_path / "metadata.yml"


@dataclass
class EstimationVolume:
    
    @dataclass
    class Metadata(YAMLObject):
        
        yaml_tag = "EstimationVolume.Metadata"
        
        exec_time: int = None
        exec_name: str = None

    volume_name: str
    volume_version: str
    model_name: str
    model_version: str
    partition: Optional[SetPartition] = None
    _metadata: Optional["EstimationVolume.Metadata"] = None

    @property
    def _volume(self) -> Volume:
        return Volume(self.volume_name, self.volume_version)

    @property
    def fullname(self) -> str:
        if self.partition is None:
            partition_name = "whole-volume"
        else:
            partition_name = self.partition.alias
            if partition_name is None:
                partition_name = f"x:{self.partition.x_range[0]}:{self.partition.x_range[1]}-y:{self.partition.y_range[0]}:{self.partition.y_range[1]}-z:{self.partition.z_range[0]}:{self.partition.z_range[1]}"
        return f"vol={self._volume.fullname}.set={partition_name}.model={self.model_name}"

    @property
    def metadata_path(self) -> Path:
        return self._volume.dir / f"{self.fullname}.metadata.yml"

    @property
    def metadata(self) -> "EstimationVolume.Metadata":

        if not self.metadata_path.exists():
            logger.debug(f"Creating metadata file at {self.metadata_path} {(now := datetime.now())=}")
            self.write_metadata("create-datetime", now)

        if self._metadata is None:
            logger.debug(f"Loading metadata from `{self.metadata_path}`.")
            with self.metadata_path.open("r") as file:
                self._metadata = yaml.load(file, Loader=yaml.FullLoader)

        return self._metadata

    def write_metadata(self, key, value):
        logger.debug(f"Writing to metadata file at `{self.metadata_path}`")

        if self._metadata is None:
            self._metadata = self.Metadata()

        setattr(self._metadata, key, value)

        with self.metadata_path.open("w") as f:
            yaml.dump(self._metadata, f, default_flow_style=False, indent=4)
            
    @property
    def dir(self) -> Path:
        (dir_ := self._volume.dir / f"{self.fullname}").mkdir(exist_ok=True)
        return dir_

    @property
    def probabilities_path(self) -> Path:
        return self.dir / f"{self.fullname}.probabilities.npy"

    def get_class_probability_path(self, class_idx: int) -> Path:
        assert class_idx in self._volume.metadata.labels
        return self.dir / f"{self.fullname}.probability.class_idx={class_idx}.raw"

    @property
    def predictions_path(self) -> Path:
        return self.dir / f"{self.fullname}.predictions.raw"

    @property
    def presoftmax_pixel_embeddings_path(self) -> Path:
        return self.dir / f"{self.fullname}.presoftmax_pixel_embeddings.npy"
    
    @property
    def pixelwise_classification_report_human(self) -> Path:
        return self.dir / f"{self.fullname}.classification_report.human.yaml"

    @property
    def pixelwise_classification_report_exact(self) -> Path:
        return self.dir / f"{self.fullname}.classification_report.exact.yaml"
    
    def get_class_roc_curve_path(self, class_idx: int) -> Path:
        assert class_idx in self._volume.metadata.labels
        return self.dir / f"{self.fullname}.roc_curve.class_idx={class_idx}.raw"

    @property
    def binary_confusion_matrices_path(self) -> Path:
        return self.dir / f"{self.fullname}.binary_confusion_matrices.npy"

    @property
    def confusion_matrix_path(self) -> Path:
        return self.dir / f"{self.fullname}.confusion_matrix.npy"

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
        return self.dir / f"{self.fullname}.confusion-volume.class_idx={class_idx}.raw"
    
    @classmethod
    def from_objects(cls, volume: Volume, model: ModelPaths, set_partition: SetPartition = None):
        return cls(volume.name, volume.version, model.name, model.version, partition=set_partition)
