from collections import namedtuple
import pathlib
from pathlib import Path

from .logger import logger

here = pathlib.Path(__file__).parent.resolve().absolute()
root_dir = (here / "..").resolve()
data_dir = (root_dir / "data").resolve()
models_dir = (root_dir / "models").resolve()

data_dir.mkdir(exist_ok=True)
models_dir.mkdir(exist_ok=True)


class VolumePaths(namedtuple("VolumePaths", ["volume_name"])):

    @property
    def train_data_path(self) -> Path:
        return data_dir / f"train.{self.volume_name}.raw"

    @property
    def train_info_path(self) -> Path:
        return data_dir / f"train.{self.volume_name}.raw.info"

    @property
    def train_labels_path(self) -> Path:
        return data_dir / f"train.labels.{self.volume_name}.raw"

    @property
    def train_weights_path(self) -> Path:
        return data_dir / f"train.weights.{self.volume_name}.raw"

    @property
    def val_data_path(self) -> Path:
        return data_dir / f"val.{self.volume_name}.raw"

    @property
    def val_info_path(self) -> Path:
        return data_dir / f"val.{self.volume_name}.raw.info"

    @property
    def val_labels_path(self) -> Path:
        return data_dir / f"val.labels.{self.volume_name}.raw"

    @property
    def val_weights_path(self) -> Path:
        return data_dir / f"val.weights.{self.volume_name}.raw"

    @property
    def test_data_path(self) -> Path:
        return data_dir / f"test.{self.volume_name}.raw"

    @property
    def test_info_path(self) -> Path:
        return data_dir / f"test.{self.volume_name}.raw.info"

    @property
    def test_labels_path(self) -> Path:
        return data_dir / f"test.labels.{self.volume_name}.raw"

    @property
    def test_weights_path(self) -> Path:
        return data_dir / f"test.weights.{self.volume_name}.raw"

    @classmethod
    def with_check(cls, volume_name):
        vol_paths: "VolumePaths" = cls(volume_name=volume_name)

        # the minimal files required
        error_paths = [
            # train
            vol_paths.train_data_path,
            vol_paths.train_labels_path,
            vol_paths.train_info_path,

            # validation
            vol_paths.val_data_path,
            vol_paths.val_labels_path,
            vol_paths.val_info_path,
        ]

        # these are not essential but important
        warning_paths = [
            # train
            vol_paths.train_weights_path,

            # validation
            vol_paths.val_weights_path,

            # test
            vol_paths.test_data_path,
            vol_paths.test_labels_path,
            vol_paths.test_info_path,
            vol_paths.test_weights_path,
        ]

        for p in error_paths:
            logger.error("Missing file: %s", str(p))

        for p in warning_paths:
            logger.warning("Missing file: %s", str(p))


# VOLUME NAMES

#    precipitates dryrun
VOLUME_PRECIPITATES_DRYRUN = "PA66GF30_trans3_x__0_pag"

#    precipitates
VOLUME_PRECIPITATES_V1 = "PA66GF30"


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
