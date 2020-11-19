from dataclasses import dataclass
from pathlib import Path

from .data import models_dir


@dataclass
class Model:

    master_name: str
    version: str
    fold: int
    runid: int  #

    @property
    def fullname(self) -> str:
        return f"{self.master_name}.{self.version}.{self.runid:03d}.{f'{(s := str(self.runid))[:4]}-{s[4:7]}-{s[7:]}'}"

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