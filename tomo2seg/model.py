import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from tomo2seg import utils
from .data import models_dir as MODELS_DIR


def models_dir(master_name: str) -> Path:
    return (MODELS_DIR / master_name).absolute()


@dataclass
class Model:

    master_name: str
    version: str

    fold: int = 0
    runid: int = field(default_factory=lambda: int(time.time()))

    factory_function: Optional[str] = None
    factory_kwargs: Optional[dict] = None

    @classmethod
    def build_from_model_name(cls, name: str):
        master_name, version, fold_str, runid_str = name.split(".")
        fold = int(fold_str.split("fold")[1])
        runid = utils.parse_runid(runid_str)
        return cls(
            master_name, version, fold, runid
        )

    def __post_init__(self):
        if self.factory_function is not None and callable(self.factory_function):
            self.factory_function = f"{self.factory_function.__module__}.{self.factory_function.__name__}"

    @staticmethod
    def vars2name(master_name: str, version: str, fold: int, runid: int):
        return Model.name_pieces2name(
            master_name=master_name,
            version=version,
            fold_str=f"fold{fold:03d}",
            runid_str=utils.fmt_runid(runid),
        )

    @staticmethod
    def name_pieces2name(master_name: str, version: str, fold_str: str, runid_str: str):
        return f"{master_name}.{version}.{fold_str}.{runid_str}"

    @property
    def name(self) -> str:
        return Model.vars2name(
            self.master_name,
            self.version,
            self.fold,
            self.runid,
        )

    @property
    def model_path(self) -> Path:
        return models_dir(master_name=self.master_name) / f"{self.name}"

    @property
    def model_path_str(self) -> str:
        return str(self.model_path)

    @property
    def autosaved_model_path(self) -> Path:
        return models_dir(master_name=self.master_name) / f"{self.name}.autosaved.hdf5"

    @property
    def autosaved_model_path_str(self) -> str:
        return str(self.autosaved_model_path)

    @property
    def autosaved2_model_path(self) -> Path:
        return self.model_path / f"{self.name}.autosaved.{{epoch:03d}}-{{val_loss:.6f}}.hdf5"

    @property
    def autosaved2_model_path_str(self) -> str:
        return str(self.autosaved2_model_path)
    
    @property
    def _autosaved2_model_filename_regex_expression(self) -> str:
        return f"{self.name}.autosaved.\d{{3,}}-(0.\d{{6,}}).hdf5".replace(".", "\.")

    @property
    def autosaved2_best_model_path(self) -> Path:
        
        is_autosaved_model = re.compile(self._autosaved2_model_filename_regex_expression)
        
        autosaved_models = []

        for filename in os.listdir(self.model_path):

            if (match := is_autosaved_model.match(filename)):

                autosaved_models.append((match.group(), match.groups()[0]))
        
        return self.model_path / min(autosaved_models, key=lambda x: x[1])[0]

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

    @property
    def train_metacrop_history_path(self) -> Path:
        return self.model_path / "metacrop-history.csv"

    @property
    def train_history_plot_wip_path(self) -> Path:
        return self.model_path / "train-hist-plot-wip.png"

    @property
    def train_log_path(self) -> Path:
        return self.model_path / f"{self.name}.train.log"
