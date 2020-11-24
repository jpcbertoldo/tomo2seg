from pathlib import Path
from typing import Optional

import pandas
from attr import dataclass
from tensorflow.keras.callbacks import History as KerasHistory
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras import backend as K

from .logger import logger


class History(KerasHistory):

    def __init__(
        self,
        optimizer: Optional[Optimizer],
        backup: Optional[int] = None,
        csv_path: Optional[Path] = None
    ):
        """

        :param optimizer: if given then the learning rate will be tracked as well
        :param backup: if given, every `backup` epochs the history is saved - in this case `csv_path` must be given
        :type csv_path: where the history will be saved
        """
        super().__init__()

        if backup is not None:
            assert backup > 0
            assert csv_path is not None, "If you want to backup you better tell me where."
            csv_path.touch()

        self.backup = backup
        self.csv_path = csv_path
        self.optimizer = optimizer

    @property
    def last_epoch(self) -> int:
        return max(self.history.get("epoch", [0]))

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            other_logs = {}

            if self.optimizer is not None:
                other_logs["lr"] = K.eval(self.optimizer.lr)

            logs.update({
                "epoch": epoch,
                **other_logs,
            })

        super().on_epoch_end(epoch, logs=logs)

        if self.backup is not None and (epoch % self.backup == self.backup - 1):
            # todo make this more efficient instead of recreating the whole thing every time
            logger.info(f"Saving backup of the training history {epoch=} {self.csv_path=}")
            self.dataframe.to_csv(self.csv_path)

    @property
    def dataframe(self) -> pandas.DataFrame:
        return pandas.DataFrame(self.history).set_index("epoch")
