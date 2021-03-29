import time
from pathlib import Path
from typing import Optional

import pandas
from dataclasses import dataclass
from tensorflow.keras.callbacks import History as KerasHistory
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras import backend as K
from tensorflow.python.keras.callbacks import Callback
from matplotlib import pyplot as plt

from . import viz
from .logger import logger
from .volume_sequence import VolumeCropSequence


class History(KerasHistory):

    def __init__(
        self,
        crop_seq_train: VolumeCropSequence,
        crop_seq_val: VolumeCropSequence,
        optimizer: Optional[Optimizer],
        backup: Optional[int] = None,
        csv_path: Optional[Path] = None,
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
        self.crop_seq_train = crop_seq_train
        self.crop_seq_val = crop_seq_val
        
        self.last_log_timestamp = None
        
        if self.csv_path is not None:
            logger.info(f"Loading history from csv {self.csv_path=}.")
            try:
                history_df = pandas.read_csv(self.csv_path)
                self.history = history_df.to_dict(orient="list")
                
                from ast import literal_eval
                self.history["train.crop_shape"] = [
                    literal_eval(x) if isinstance(x, str) else x
                    for x in self.history["train.crop_shape"]
                ]

                self.history["val.crop_shape"] = [
                    literal_eval(x) if isinstance(x, str) else x
                    for x in self.history["val.crop_shape"]
                ]

            except FileNotFoundError:
                logger.debug("History hasn't been saved yet.")

            except pandas.errors.EmptyDataError:
                logger.debug("History hasn't been saved yet.")

    @property
    def last_epoch(self) -> int:
        return max(self.history.get("epoch", [0]))

    def on_train_begin(self, logs=None):
        super(History, self).on_train_begin(logs=logs)
        self.last_log_timestamp = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            other_logs = {}

            if self.optimizer is not None:
                other_logs["lr"] = K.eval(self.optimizer.lr)

            other_logs["train.batch_size"] = self.crop_seq_train.batch_size
            other_logs["train.epoch_size"] = self.crop_seq_train.epoch_size
            other_logs["train.crop_shape"] = self.crop_seq_train.crop_shape

            other_logs["val.batch_size"] = self.crop_seq_val.batch_size
            other_logs["val.epoch_size"] = self.crop_seq_val.epoch_size
            other_logs["val.crop_shape"] = self.crop_seq_val.crop_shape

            now = time.time()
            other_logs["seconds"] = now - self.last_log_timestamp
            self.last_log_timestamp = now

            logs = {
                **logs,
                "epoch": epoch,
                **other_logs,
            }

        super().on_epoch_end(epoch, logs=logs)

        if self.backup is not None and (epoch % self.backup == self.backup - 1):
            # todo make this more efficient instead of recreating the whole thing every time
            logger.info(f"Saving backup of the training history {epoch=} {self.csv_path=}")
            self.dataframe.to_csv(self.csv_path)

    @property
    def dataframe(self) -> pandas.DataFrame:
        if len(self.history) == 0:
            return pandas.DataFrame()
        return pandas.DataFrame(self.history).set_index("epoch")


@dataclass
class HistoryPlot(Callback):

    history_callback: History
    save_path: Path

    def on_epoch_end(self, epoch, logs=None):
        if epoch < 2:
            logger.debug(f"{epoch=} is too early to plot something.")

        try:

            fig, axs = plt.subplots(
                nrows := 2,
                ncols := 1,
                figsize=(2.5 * ncols * (sz := 5), nrows * sz),
                dpi=100
            )
            fig.set_tight_layout(True)

            hist_display = viz.TrainingHistoryDisplay(
                self.history_callback.history,
                x_axis_mode=(
                    "epoch",
                    "batch",
                    "crop",
                    "voxel",
                    "time",
                ),
            ).plot(
                axs,
                with_lr=True,
                metrics=("loss", ),
            )

            axs[0].set_yscale("log")
            axs[-1].set_yscale("log")

            viz.mark_min_values(hist_display.axs_metrics_[0], hist_display.plots_["loss"][0])
            viz.mark_min_values(hist_display.axs_metrics_[0], hist_display.plots_["val_loss"][0],
                                txt_kwargs=dict(rotation=0))

            hist_display.fig_.savefig(
                self.save_path,
                format='png',
            )
            plt.close()

        except Exception as ex:
            logger.exception(f"{ex.__class__.__name__} occurred while trying to plot the history.")

