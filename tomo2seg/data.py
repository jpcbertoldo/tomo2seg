import time
import warnings
from datetime import datetime
from dataclasses import dataclass, field
import pathlib
from pathlib import Path
from typing import List, Tuple, Dict, Optional, ClassVar

import yaml
from numpy import ndarray
from yaml import YAMLObject

from . import utils
from .logger import logger


here = pathlib.Path(__file__).parent.resolve().absolute()
root_dir = (here / "..").resolve()
data_dir = (root_dir / "data").resolve()
models_dir = (data_dir / "models").resolve()

data_dir.mkdir(exist_ok=True)
models_dir.mkdir(exist_ok=True)


NORMALIZE_FACTORS = {
    "uint8": 255,
    "uint16": 65535,
}


@dataclass
class SetPartition(YAMLObject):
    x_range: Tuple[int, int]
    y_range: Tuple[int, int]
    z_range: Tuple[int, int]
    alias: str

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
        if volume.ndim == 3:
            return volume[
                self.x_range[0]:self.x_range[1],
                self.y_range[0]:self.y_range[1],
                self.z_range[0]:self.z_range[1],
            ]
        elif volume.ndim == 4:
            return volume[
               self.x_range[0]:self.x_range[1],
               self.y_range[0]:self.y_range[1],
               self.z_range[0]:self.z_range[1],
               :
            ]
        else:
            raise ValueError(f"{volume.ndim=}")

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
    """"""

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

    @staticmethod
    def name_pieces2fullname(name: str, version: Optional[str]) -> str:
        if version is not None:
            return f"{name}.{version}"
        else:
            return f"{name}"

    @property
    def fullname(self) -> str:
        return Volume.name_pieces2fullname(
            name=self.name,
            version=self.version,
        )

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
    def set_partitions(self) -> Dict[str, dict]:
        warnings.warn(f"{self.__class__.__name__}.set_partitions is deprecated, please use __get_item__.", DeprecationWarning)
        return self.metadata.set_partitions

    def __getitem__(self, partition_alias) -> SetPartition:
        try:
            partition_dict = self.metadata.set_partitions[partition_alias]

        except KeyError as ex:

            if ex.args[0] != partition_alias:
                raise ex

            raise KeyError(f"{partition_alias=} not available. Pick one from {list(self.metadata.set_partitions.keys())}")

        except TypeError as ex:

            if ex.args[0] != "'NoneType' object is not subscriptable":
                raise ex

            raise KeyError(f"No partitions were defined for volume={self.fullname}")

        return SetPartition.from_dict(partition_dict)

    @property
    def train_partition(self) -> SetPartition:
        return SetPartition.from_dict(
            self.metadata.set_partitions[self.Metadata.TRAIN_PARTITION_KEY]
        )

    @property
    def val_partition(self) -> SetPartition:
        return SetPartition.from_dict(
            self.metadata.set_partitions[self.Metadata.VAL_PARTITION_KEY]
        )

    @property
    def test_partition(self) -> SetPartition:
        return SetPartition.from_dict(
            self.metadata.set_partitions[self.Metadata.TEST_PARTITION_KEY]
        )

    @classmethod
    def with_check(cls, name, version: str = None):
        vol: "Volume" = cls(name=name, version=version)

        logger.debug(f"{vol=}")

        # the minimal files required
        error_paths: List[Path] = [
            vol.data_path,
            vol.info_path,
            vol.metadata_path,
        ]

        # these are not essential but important
        warning_paths: List[Path] = [
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

    def grid_position_probabilities_path(self, partition: SetPartition, crop_shape: Tuple[int, int, int], version: str) -> Path:
        return self.dir / f"{self.fullname}.grid-position-probabilities.partition={partition.alias}.crop-shape={crop_shape}.version={version}.npy"

    @classmethod
    def from_fullname(cls, volume_fullname: str):
        name, version = volume_fullname.split(".")
        return cls(name=name, version=version)
    
    @property
    def nclasses(self) -> int:
        return len(self.metadata.labels)
    
    @property
    def normalization_factor(self) -> int:
        return NORMALIZE_FACTORS[self.metadata.dtype]


@dataclass
class EstimationVolume(YAMLObject):
    """"""

    WHOLE_VOLUME_ALIAS: ClassVar[str] = "whole-volume"

    yaml_tag: ClassVar[str] = "tomo2seg.EstimationVolume"

    volume_fullname: str
    model_name: str
    runid: int = field(default_factory=lambda: int(time.time()))
    partition: Optional[SetPartition] = None

    @property
    def fullname(self) -> str:
        partition_name = self.WHOLE_VOLUME_ALIAS if self.partition is None else (
            self.partition.alias or self.partition.canonical_alias
        )
        runid = utils.fmt_runid(self.runid)
        return f"vol={self.volume_fullname}.set={partition_name}.model={self.model_name}.runid={runid}"

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
    
    @property
    def notable_slices_path(self) -> Path:
        return self.dir / f"{self.fullname}.notable-slices.yaml"
    
    @property
    def notable_slices_plots(self) -> Path:
        return self.dir / f"notable-slices-plots"

    @property
    def classification_report_table_exact_csv_path(self) -> Path:
        return self.dir / f"{self.fullname}.classification-report-table.exact.csv"

    @property
    def classification_report_table_human_simple_txt_path(self) -> Path:
        return self.dir / f"{self.fullname}.classification-report-table.human.simple.txt"

    @property
    def classification_report_table_human_detail_txt_path(self) -> Path:
        return self.dir / f"{self.fullname}.classification-report-table.human.detail.txt"

    def get_class_roc_curve_path(self, class_idx: int) -> Path:
        return self.dir / f"{self.fullname}.roc-curve.class-idx={class_idx}.raw"

    def get_roc_curve_csv_path(self, class_idx: int) -> Path:
        return self.dir / f"{self.fullname}.roc-curve.class-idx={class_idx}.csv"

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
    
    @property
    def confusion_volume_path(self) -> Path:
        return self.dir / f"{self.fullname}.confusion-volume.raw"

    @property
    def error_volume_path(self) -> Path:
        return self.dir / f"{self.fullname}.error-volume.raw"

    @property
    def error_2dblobs_props_path(self) -> Path:
        return self.dir / f"{self.fullname}.error-2dblobs-props.csv"
    
    @property
    def exec_log_path(self) -> Path:
        return self.dir / f"{self.fullname}.exec.log"
    
    @property
    def exec_log_path_str(self) -> str:
        return str(self.exec_log_path)

    @property
    def analyse_exec_log_path(self) -> Path:
        return self.dir / f"{self.fullname}.analyse-exec.log"
    
    @classmethod
    def from_objects(cls, volume: Volume, model: "Model", set_partition: SetPartition = None, runid=None):
        return cls(
            volume_fullname=volume.fullname,
            model_name=model.name,
            partition=set_partition,
            runid=runid
        )

    @classmethod
    def from_fullname(cls, full_name: str):
        from .model import Model

        try:
            vol_name, vol_version, partition_name, model_master_name, model_version, model_fold, model_runid, runid = full_name.split(".")
            
        except ValueError as ex:
            
            logger.exception(ex)
            
            if "not enough values to unpack" not in ex.args[0]:
                raise ex
            
            raise ValueError(f"not an estimation volume {full_name=}")
            
        vol_name = vol_name.split("=")[1]
        partition_name = partition_name.split("=")[1]
        model_master_name = model_master_name.split("=")[1]
        runid = runid.split("=")[1]

        volume_fullname = Volume.name_pieces2fullname(
            name=vol_name,
            version=vol_version,
        )

        model_name = Model.name_pieces2name(
            master_name=model_master_name,
            version=model_version,
            fold_str=model_fold,
            runid_str=model_runid
        )

        assert partition_name in (
            cls.WHOLE_VOLUME_ALIAS,
            Volume.Metadata.TRAIN_PARTITION_KEY,
            Volume.Metadata.VAL_PARTITION_KEY,
            Volume.Metadata.TEST_PARTITION_KEY,
        ), f"{partition_name=}"

        runid = utils.parse_runid(runid)

        if partition_name == "whole-volume":
            partition = None

        else:
            logger.info("Creating volume object to get partition dimensions.")

            volume = Volume.from_fullname(volume_fullname)
            partition = SetPartition.from_dict(volume.set_partitions.get(partition_name))

        return cls(
            volume_fullname=volume_fullname,
            model_name=model_name,
            runid=runid,
            partition=partition
        )
