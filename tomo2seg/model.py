import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .data import models_dir


@dataclass
class Model:

    master_name: str
    version: str

    fold: int = 0
    runid: int = field(default_factory=lambda: int(time.time()))

    factory_function: Optional[str] = None
    factory_kwargs: Optional[dict] = None

    def __post_init__(self):
        if self.factory_function is not None and callable(self.factory_function):
            self.factory_function = f"{self.factory_function.__module__}.{self.factory_function.__name__}"

    @property
    def name(self) -> str:
        s = str(self.runid)
        return f"{self.master_name}.{self.version}.{self.fold:03d}.{f'{s[:4]}-{s[4:7]}-{s[7:]}'}"

    @property
    def model_path(self) -> Path:
        return models_dir / f"{self.name}"

    @property
    def model_path_str(self) -> str:
        return str(self.model_path)

    @property
    def autosaved_model_path(self) -> Path:
        return models_dir / f"{self.name}.autosaved.hdf5"

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
