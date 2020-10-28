from collections import namedtuple
import pathlib
from dataclasses import dataclass
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

    def get_volume_partition(self, volume: ndarray) -> ndarray:
        return volume[
            self.x_range[0]:self.x_range[1],
            self.y_range[0]:self.y_range[1],
            self.z_range[0]:self.z_range[1],
       ]


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

    TRAIN_PARTITION_KEY = "train"
    VAL_PARTITION_KEY = "val"
    TEST_PARTITION_KEY = "test"

    volume_name: str
    version_name: Optional[str] = None
    _set_partitions: Dict[str, SetPartition] = None

    @property
    def fullname(self) -> str:
        if self.version_name is not None:
            return f"{self.volume_name}.{self.version_name}"
        else:
            return f"{self.volume_name}"

    @property
    def dir(self) -> Path:
        return data_dir / f"{self.fullname}"

    @property
    def data_path(self) -> Path:
        return self.dir / f"{self.fullname}.raw"

    @property
    def info_path(self) -> Path:
        return self.dir / f"{self.fullname}.raw.info"

    @property
    def labels_path(self) -> Path:
        return self.dir / f"{self.fullname}.labels.raw"

    @property
    def weights_path(self) -> Path:
        return self.dir / f"{self.fullname}.weights.raw"

    @property
    def set_partitions_path(self) -> Path:
        return self.dir / f"{self.fullname}.set-partitions.yml"

    @property
    def set_partitions(self) -> Dict[str, SetPartition]:
        if self._set_partitions is None:
            with self.set_partitions_path.open("r") as file:
                self._set_partitions = yaml.load(file)
        return self._set_partitions

    @property
    def train_partition(self) -> SetPartition:
        return self.set_partitions[self.TRAIN_PARTITION_KEY]

    @property
    def val_partition(self) -> SetPartition:
        return self.set_partitions[self.VAL_PARTITION_KEY]

    @property
    def test_partition(self) -> SetPartition:
        return self.set_partitions[self.TEST_PARTITION_KEY]

    @classmethod
    def with_check(cls, volume_name, version_name: str = None):
        vol: "Volume" = cls(volume_name=volume_name, version_name=version_name)

        logger.debug(f"{vol=}")

        # the minimal files required
        error_paths: List[Path] = [
            vol.data_path,
            vol.labels_path,
            vol.info_path,
        ]

        # these are not essential but important
        warning_paths: List[Path] = [
            # train
            vol.weights_path,
            vol.set_partitions_path,
        ]

        for p in error_paths:
            if not p.exists():
                logger.error("Missing file: %s", str(p))

        for p in warning_paths:
            if not p.exists():
                logger.warning("Missing file: %s", str(p))

        if vol.set_partitions_path.exists():

            # train/val partitions must exist
            for key in [cls.TRAIN_PARTITION_KEY, cls.VAL_PARTITION_KEY]:
                if key not in vol.set_partitions:
                    logger.error("Missing set partition: %s", str(key))

            # test partition SHOULD...
            for key in [cls.TEST_PARTITION_KEY]:
                if key not in vol.set_partitions:
                    logger.error("Missing set partition: %s", str(key))

        return vol


@dataclass
class EstimationVolume:

    volume: Volume
    partition: SetPartition
    model_name: str

    @property
    def fullname(self) -> str:
        partition_name = self.partition.alias
        if partition_name is None:
            partition_name = f"x:{self.partition.x_range[0]}:{self.partition.x_range[1]}-y:{self.partition.y_range[0]}:{self.partition.y_range[1]}-z:{self.partition.z_range[0]}:{self.partition.z_range[1]}"
        return f"vol:{self.volume.fullname}--set:{partition_name}--model:{self.model_name}"

    @property
    def probabilities_path(self) -> Path:
        return self.volume.dir / f"{self.fullname}.probabilities.raw"

    @property
    def predictions_path(self) -> Path:
        return self.volume.dir / f"{self.fullname}.predictions.raw"


# VOLUME NAMES / VERSIONS

#    precipitates dryrun
VOLUME_PRECIPITATES_DRYRUN = "PA66GF30_trans3_x__0_pag"

#    precipitates
VOLUME_PRECIPITATES_V1 = "PA66GF30", "v1"


class ModelPaths(namedtuple("ModelPaths", ["model_name"])):

    @property
    def model_path(self) -> Path:
        return models_dir / f"{self.model_name}"

    @property
    def autosaved_model_path(self) -> Path:
        return models_dir / f"{self.model_name}-autosaved"

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
