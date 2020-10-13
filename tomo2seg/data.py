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
    def train_data_path(self) -> Path:
        return data_dir / f"train.{self.volume_name}.raw"

    def train_info_path(self) -> Path:
        return data_dir / f"train.{self.volume_name}.raw.info"

    def train_labels_path(self) -> Path:
        return data_dir / f"train.labels.{self.volume_name}.raw"

    def train_weights_path(self) -> Path:
        return data_dir / f"train.weights.{self.volume_name}.raw"

    def val_data_path(self) -> Path:
        return data_dir / f"val.{self.volume_name}.raw"

    def val_info_path(self) -> Path:
        return data_dir / f"val.{self.volume_name}.raw.info"

    def val_labels_path(self) -> Path:
        return data_dir / f"val.labels.{self.volume_name}.raw"

    def val_weights_path(self) -> Path:
        return data_dir / f"val.weights.{self.volume_name}.raw"


# VOLUMES
VOLUME_PRECIPITATES_DRYRUN = "PA66GF30_trans3_x__0_pag"
volume_precipitates_dryrun = VolumePaths(VOLUME_PRECIPITATES_DRYRUN)


class ModelPaths(namedtuple("ModelPaths", ["model_name"])):

    def model_path(self) -> Path:
        return models_dir / f"{self.model_name}"

    def autosaved_model_path(self) -> Path:
        return models_dir / f"{self.model_name}-autosaved"

    def logger_path(self) -> Path:
        return self.model_path() / "logger.csv"

    def history_path(self) -> Path:
        return self.model_path() / "history.csv"

    def summary_path(self) -> Path:
        return self.model_path() / "summary.txt"

    def architecture_plot_path(self) -> Path:
        return self.model_path() / "architecture.png"

