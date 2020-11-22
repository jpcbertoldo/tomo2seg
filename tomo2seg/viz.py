#
import copy
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from string import Template
from typing import List, Optional, Union, Dict, Tuple

import humanize
from matplotlib import pyplot as plt
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.pyplot import Figure, Axes, cm
import numpy as np
from matplotlib.text import Text
from numpy import ndarray
from sklearn.utils import check_matplotlib_support
from tomo2seg.callbacks import History

from .logger import logger


FIGS_COMMON_METADATA = dict(
    Author="joaopcbertoldo",
    Software="tomo2seg",
    Location="Paris/France",
    Credits="Bigméca, Centre des Matériaux Mines ParisTech, École des Mines de Paris",
)


def tight_subplots(n: int, m: int, w: float, h: float) -> (Figure, Axes):
    fig, axs = plt.subplots(n, m, figsize=(w, h), sharex='all', sharey='all')
    fig.set_tight_layout(True)
    return fig, axs


def plot_orthogonal_slices(axs: ndarray, volume: ndarray, labels_mask: ndarray = None, normalized_voxels=False, is_color=False, vrange=None):
    logger.debug(f"{volume.shape=}")

    vmin, vmax = (0, 1) if normalized_voxels else (0, 255) if vrange is None else vrange

    logger.debug(f"{vmin, vmax=}")

    if labels_mask is not None:
        assert volume.shape == labels_mask.shape
    else:
        logger.debug("No label mask given.")

    xy_axis, yz_axis, xz_axis = axs[0, 0], axs[0, 1], axs[1, 0]
    axs[1, 1].axis(False)

    if is_color:
        (height, width, depth, _) = volume.shape
    else:
        (height, width, depth) = volume.shape

    xy_z_coord, yz_x_coord, xz_y_coord = int(depth // 2), int(width // 2), int(height // 2)

    logger.debug(f"{xy_z_coord, yz_x_coord, xz_y_coord=}")

    kwargs = dict(interpolation=None) if is_color else dict(vmin=vmin, vmax=vmax, cmap=cm.gray, interpolation=None)
    xy_axis.imshow(xy_slice := volume[:, :, xy_z_coord], **kwargs)
    yz_axis.imshow(yz_slice := volume[yz_x_coord, :, :], **kwargs)
    xz_axis.imshow(np.rot90(xz_slice := volume[:, xz_y_coord, :]), **kwargs)

    logger.debug(f"{xy_slice.shape, yz_slice.shape, xz_slice.shape=}")

    if labels_mask is not None:
        kwargs = dict(cmap=cm.inferno, interpolation=None, alpha=0.5)
        volume_masked = np.ma.masked_where(labels_mask, volume)
        xy_axis.imshow(volume_masked[:, :, xy_z_coord], **kwargs)
        yz_axis.imshow(volume_masked[yz_x_coord, :, :], **kwargs)
        xz_axis.imshow(volume_masked[:, xz_y_coord, :], **kwargs)

    xy_axis.set_title(f"XY :: z={xy_z_coord}")
    yz_axis.set_title(f"YZ :: x={yz_x_coord}")
    xz_axis.set_title(f"XZ :: y={xz_y_coord}")
    for ax in axs.ravel():
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    # todo generalize this so that I can plot slices from any coordinates --> pick randomly in usage


def show_slice(ax: Axes, slice_: ndarray, labels_mask: ndarray = None):

    if labels_mask is not None:
        assert slice_.shape == labels_mask.shape

    ax.imshow(slice_, cmap=cm.gray, interpolation=None)
    if labels_mask is not None:
        slice_masked = np.ma.masked_where(labels_mask, slice_)
        ax.imshow(slice_masked, cmap=cm.inferno, interpolation=None, alpha=0.5)


def display_training_curves(training, validation, title, subplot, x=None):
    ax = plt.subplot(subplot)
    if x is None:
        ax.plot(training)
        ax.plot(validation)
    else:
        # todo clean up this function
        ax.plot(x, training)
        ax.plot(x, validation)
    ax.set_title('model '+ title)
    ax.set_ylabel(title)
    ax.set_xlabel('epoch')
    ax.legend(['training', 'validation'])


@dataclass
class Display(ABC):

    fig_: Figure = field(init=False)
    axs_: Union[Axes, ndarray] = field(init=False)
    plots_: Dict[str, Union[List[Line2D], AxesImage]] = field(init=False, default_factory=dict)
    creation_time: int = field(init=False, default_factory=lambda: int(time.time()))

    @abstractmethod
    def plot(self) -> "Display":
        pass

    @property
    @abstractmethod
    def title(self) -> str:
        pass

    @property
    def _class_metadata(self):
        return {}

    @property
    def metadata(self) -> dict:
        return {
            **FIGS_COMMON_METADATA,
            **dict(
                CreationTimeISO=datetime.fromtimestamp(self.creation_time).isoformat(timespec='seconds'),
                Title=self.title,
            ),
            **self._class_metadata,
        }


@dataclass
class TrainingHistoryDisplay(Display):
    """Structured inspired in `sklearn.metrics.RocCurveDisplay`"""

    history: Dict[str, List]
    model_name: Optional[str] = None
    loss_name: Optional[str] = None

    # not arguments
    ax_loss_: Axes = field(init=False)
    ax_lr_: Axes = field(init=False)

    @property
    def title(self) -> str:
        return (self.model_name or "unknown-model") + ".training-history-plot"

    def plot(
        self,
        axs=None,
        with_lr: bool = False,
        loss_kwargs: dict = None,
        val_loss_kwargs: dict = None,
        lr_kwargs: dict = None
    ) -> "TrainingHistoryDisplay":
        check_matplotlib_support(this_func_name := f"{(this_class_name := self.__class__.__name__)}.plot")

        # noinspection PyShadowingNames
        import matplotlib.pyplot as plt

        def _missing_signal_error_msg(key: str, keys=False) -> str:
            msg = f"The history dict given to {this_func_name} does not have the *{key}*."
            if keys:
                msg += f"\n{list(self.history.keys())=}"
            return msg

        if axs is None:
            if with_lr:
                fig, axs = plt.subplots(2, 1)
            else:
                fig, axs = plt.subplots(1, 1)

        if with_lr:
            try:
                ax_loss: Axes = axs[0]
                ax_lr: Axes = axs[1]
                self.ax_lr_ = ax_lr

            except TypeError as ex:
                if "'AxesSubplot' object is not subscriptable" not in ex.args[0]:
                    raise ex

                logger.error(f"Trying to plot a {this_class_name} with {with_lr=} with a single ax.")
                raise ValueError(f"The argument `axs` should be a 1d-array of axes.")
        else:
            assert isinstance(axs, Axes)
            ax_loss = axs

        # i don't know why this is done, I just copied
        self.axs_ = axs
        self.fig_ = ax_loss.figure
        self.ax_loss_ = ax_loss

        try:
            epoch = self.history["epoch"]

        except KeyError as ex:

            if ex.args[0] != "epoch":
                raise ex

            n_epochs = len(self.history["loss"])
            logger.warning(
                f"{_missing_signal_error_msg('epoch', True)}\n"
                f"Using a default sequence (0, 1, ..., {n_epochs - 1=})"
            )
            epoch = np.arange(0, n_epochs)

        assert len(epoch) > 1, "You don't have enough epochs to plot. Go to the gym and call me later."

        try:
            loss = self.history["loss"]

            line_loss_kwargs = dict(label="train")
            # noinspection PyArgumentList
            line_loss_kwargs.update(**(loss_kwargs or dict()))
            self.plots_["loss"] = ax_loss.plot(epoch, loss, **line_loss_kwargs)

        except KeyError as ex:
            if ex.args[0] == "loss":
                logger.error(f"{_missing_signal_error_msg('loss', True)}")
            raise ex

        try:
            val_loss = self.history["val_loss"]

            line_val_loss_kwargs = dict(label="val")
            # noinspection PyArgumentList
            line_val_loss_kwargs.update(**(val_loss_kwargs or dict()))
            self.plots_["val_loss"] = ax_loss.plot(epoch, val_loss, **line_val_loss_kwargs)

        except KeyError as ex:
            if ex.args[0] != "val_loss":
                raise ex
            logger.warning(f"{_missing_signal_error_msg('val_loss', True)}")

        ax_loss.set_title(f"Loss history {f'({self.loss_name})' or ''}")
        ax_loss.set_ylabel(f"{self.loss_name or 'loss'}")
        ax_loss.set_xlabel("epoch")

        # losses tend to go down, so this should be a good position
        # notice that using default loc=None is slower
        ax_loss.legend(loc="upper right")

        try:
            self.plots_["lr"] = ax_lr.plot(epoch, self.history["lr"], label="lr")
            ax_lr.set_title("Learning rate history")
            ax_lr.set_ylabel(f"learning rate (lr)")
            ax_lr.set_xlabel("epoch")
            ax_lr.legend()

        except UnboundLocalError as ex:
            if ex.args[0] != "local variable 'ax_lr' referenced before assignment":
                raise ex
            # simply not using the lr option

        except KeyError as ex:
            if ex.args[0] == "lr":
                logger.error(f"{_missing_signal_error_msg('lr', True)}")
            raise ex

        if self.model_name is not None:
            self.fig_.suptitle(f"model: {self.model_name}")

        return self


def mark_min_values(ax: Axes, plot: Line2D, fmt_x=".3g", fmt_y=".3g", with_txt=True) -> Union[Line2D, Tuple[Line2D, Text]]:
    xy = plot.get_xydata()
    argmin_y = int(np.argmin(xy[:, 1]))
    label = plot.get_label()
    x_argmin_y, min_y = xy[argmin_y, :]

    logger.info(
        f"{label}: argmin={x_argmin_y:{fmt_x}} --> min={min_y:{fmt_y}}"
    )

    ylim = ax.get_ylim()
    color = plot.get_color()
    vlines_plot = ax.vlines(
        x_argmin_y, min_y, ylim[1] / 2,
        linestyles='--',
        label=f"min({label})",
        colors=color,
    )

    if with_txt:
        txt = ax.text(
            x_argmin_y, ylim[1] / 2,
            f"min({label})\n(x={x_argmin_y:{fmt_x}}, y={min_y:{fmt_y}})",
            fontdict=dict(rotation=60, color=color),
            rotation_mode='anchor',
        )

    ax.set_ylim(*ylim)

    if with_txt:
        return vlines_plot, txt

    return vlines_plot


@dataclass
class OrthogonalSlicesDisplay(Display):

    volume: ndarray
    volume_name: str
    x: int = None
    y: int = None
    z: int = None
    xy_axis_: Axes = field(init=False)
    yz_axis_: Axes = field(init=False)
    xz_axis_: Axes = field(init=False)

    def __post_init__(self):
        (width, height, depth) = self.volume.shape
        self.x, self.y, self.z = self.x or int(width // 2), self.y or int(height // 2), self.z or int(depth // 2)
        assert 0 <= self.x < self.volume.shape[0], f"{self.x=} out of range {self.volume.shape[0]=})"
        assert 0 <= self.y < self.volume.shape[1], f"{self.y=} out of range {self.volume.shape[1]=})"
        assert 0 <= self.z < self.volume.shape[2], f"{self.z=} out of range {self.volume.shape[2]=})"

    @property
    def title(self) -> str:
        return f"{self.volume_name}.orthogonal-slices-display.x={self.x}-y={self.y}-z={self.z}"

    def plot(self, axs: ndarray = None, with_cuts=True, imshow_kwargs: dict = None,
             cut_lines_kwargs: dict = None) -> "OrthogonalSlicesDisplay":
        check_matplotlib_support(this_func_name := f"{(this_class_name := self.__class__.__name__)}.plot")

        # noinspection PyShadowingNames
        import matplotlib.pyplot as plt

        if axs is None:
            fig, axs = plt.subplots(2, 2)
        else:
            fig: Figure = axs[0, 0].figure
            assert isinstance(axs, ndarray)
            assert axs.shape == (2, 2)
            assert all(isinstance(obj, Axes) for obj in axs.ravel())

        xy_axis: Axes = axs[0, 0]
        yz_axis: Axes = axs[0, 1]
        xz_axis: Axes = axs[1, 0]
        axs[1, 1].axis(False)

        self.fig_ = fig
        self.axs_ = axs
        self.xy_axis_, self.yz_axis_, self.xz_axis_ = xy_axis, yz_axis, xz_axis

        imshow_kwargs_ = {
            **dict(cmap=cm.gray, interpolation=None, vmin=0, vmax=1),
            **(imshow_kwargs or {})
        }  # override with user-given kwargs

        slice_xy = self.volume[:, :, self.z].T
        slice_yz = self.volume[self.x, :, :]
        slice_xz = self.volume[:, self.y, :].T

        # noinspection PyArgumentList
        self.plots_["xy_slice"] = xy_axis.imshow(slice_xy, **imshow_kwargs_)
        # noinspection PyArgumentList
        self.plots_["yz_slice"] = yz_axis.imshow(slice_yz, **imshow_kwargs_)
        # noinspection PyArgumentList
        self.plots_["xz_slice"] = xz_axis.imshow(slice_xz, **imshow_kwargs_)

        spines_specs_dict = dict(linewidth=4)

        xy_axis.set_xlabel("x")
        xy_axis.set_ylabel("y")
        xy_axis.set_title(f"XY :: z={self.z}", fontdict=dict(color='g'))

        yz_axis.set_xlabel("z")
        yz_axis.set_ylabel("y")
        yz_axis.set_title(f"YZ :: x={self.x}", fontdict=dict(color='r'))

        xz_axis.set_xlabel("x")
        xz_axis.set_ylabel("z")
        xz_axis.set_title(f"XZ :: y={self.y}", fontdict=dict(color='b'))

        if with_cuts:
            axes_list = [xy_axis, yz_axis, xz_axis]
            lims = [(axis.get_xlim(), axis.get_ylim()) for axis in axes_list]
            cut_lines_kwargs = {
                **dict(linestyles="--", linewidth=4),
                **(cut_lines_kwargs or {})
            }
            xy_axis.vlines(self.x, 0, slice_xy.shape[0], color='r', **cut_lines_kwargs)
            xy_axis.hlines(self.y, 0, slice_xy.shape[1], color='b', **cut_lines_kwargs)
            yz_axis.vlines(self.z, 0, slice_yz.shape[0], color='g', **cut_lines_kwargs)
            yz_axis.hlines(self.y, 0, slice_yz.shape[1], color='b', **cut_lines_kwargs)
            xz_axis.vlines(self.x, 0, slice_xz.shape[0], color='r', **cut_lines_kwargs)
            xz_axis.hlines(self.z, 0, slice_xz.shape[1], color='g', **cut_lines_kwargs)
            plt.setp(xy_axis.spines.values(), color="g", **spines_specs_dict)
            plt.setp(yz_axis.spines.values(), color="r", **spines_specs_dict)
            plt.setp(xz_axis.spines.values(), color="b", **spines_specs_dict)
            for axis, lim in zip(axes_list, lims):
                axis.set_xlim(lim[0])
                axis.set_ylim(lim[1])

        fig.suptitle(f"Orthogonal slices\nVolume: {self.volume_name}")

        return self


@dataclass
class SliceDataPredictionDisplay(Display):

    slice_data: ndarray
    slice_prediction: ndarray
    slice_name: str
    n_classes: int
    is_probability: bool = False
    class_name: Optional[str] = None

    data_axis_: Axes = field(init=False)
    pred_axis_: Axes = field(init=False)

    @property
    def title(self) -> str:
        return f"{self.slice_name}.data-prediction"

    def plot(self, axs: ndarray, data_imshow_kwargs: dict = None, pred_imshow_kwargs: dict = None) -> "SliceDataPredictionDisplay":

        check_matplotlib_support(this_func_name := f"{(this_class_name := self.__class__.__name__)}.plot")

        # noinspection PyShadowingNames
        import matplotlib.pyplot as plt

        if axs is None:
            fig, axs = plt.subplots(1, 2)
        else:
            assert isinstance(axs, ndarray)
            assert axs.shape == (2,)
            assert all(isinstance(obj, Axes) for obj in axs.ravel())
            fig: Figure = axs[0].figure

        data_axis: Axes = axs[0]
        pred_axis: Axes = axs[1]

        self.fig_ = fig
        self.axs_ = axs
        self.data_axis_ = data_axis
        self.pred_axis_ = pred_axis

        data_imshow_kwargs = {
            **dict(cmap=cm.gray, interpolation=None, vmin=0, vmax=1),
            **(data_imshow_kwargs or {})
        }  # override with user-given kwargs

        data_axis.imshow(self.slice_data, **data_imshow_kwargs)
        data_axis.set_title("data")

        pred_imshow_kwargs = {
            **dict(cmap=cm.gray, interpolation=None, vmin=0, vmax=(1 if self.is_probability else self.n_classes - 1)),
            **(pred_imshow_kwargs or {})
        }  # override with user-given kwargs

        pred_axis.imshow(self.slice_prediction, **pred_imshow_kwargs)
        pred_axis.set_title("probability" + (f" class={self.class_name}" if self.class_name is not None else "") if self.is_probability else "prediction")

        fig.suptitle(f"Data/{'probability' if self.is_probability else 'prediction'} comparison on slice={self.slice_name}")

        return self


@dataclass
class ClassImbalanceDisplay(Display):

    volume_name: str
    labels_idx: List[int]
    labels_names: List[str]
    labels_counts: List[int]
    n_voxels: int = field(init=False)

    def __post_init__(self):
        self.n_voxels = 0
        for c in self.labels_counts:
            self.n_voxels += c

    @property
    def title(self) -> str:
        return f"{self.volume_name}.class-imbalance"

    def plot(self, ax=None) -> "Display":

        check_matplotlib_support(this_func_name := f"{(this_class_name := self.__class__.__name__)}.plot")

        # noinspection PyShadowingNames
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            assert isinstance(ax, Axes)
            fig: Figure = ax.figure

        self.axs_ = ax
        self.fig_ = fig

        self.plots_["barh"] = ax.barh(self.labels_idx, self.labels_counts)
        ax.set_yticks(self.labels_idx)
        ax.set_yticklabels([f"{self.labels_names[idx]} ({idx=})" for idx in self.labels_idx])
        ax.set_xticks([])
        for idx, count in zip(self.labels_idx, self.labels_counts):
            self.plots_[f"text.label_idx={idx}"] = ax.text(
                max(self.labels_counts) // 10, idx,
                f"{humanize.intword(count)} "
                f"({humanize.intcomma(count)}) = {(perc := count / self.n_voxels):.0%} ({perc:.4%})",
                fontsize="large"
            )

        ax.set_title(f"Class imbalance of {self.volume_name}")

        random_model_accu = max(self.labels_counts) / self.n_voxels
        logger.info(f"The minimum accuracy is: {random_model_accu=:.2%}")

        return self


@dataclass
class VoxelValueHistogramDisplay(Display):

    volume_name: str
    bins: list
    values: list

    ax_log_: Axes = field(init=False)

    def __post_init__(self):
        assert len(self.bins) == 256, f"{len(self.bins)=}"
        assert len(self.values) == 256, f"{len(self.values)=}"
        assert min(self.bins) == 0, f"{min(self.bins)=}"
        assert max(self.bins) == 255, f"{max(self.bins)}"

        # i want to get the vertical borders to show up
        self.bins += [256, 257]
        self.values = [0] + self.values + [0]

    @property
    def title(self) -> str:
        return f"{self.volume_name}.voxel-value-histogram"

    def plot(self, ax: Axes, y_linear_kwargs: dict = None, y_log_kwargs: dict = None) -> "VoxelValueHistogram":
        check_matplotlib_support(this_func_name := f"{(this_class_name := self.__class__.__name__)}.plot")

        self.axs_ = ax
        self.fig_ = ax.figure
        self.ax_log_ = axlog = ax.twinx()

        y_linear_kwargs = {
            **dict(linewidth=.75, color='blue', label='linear',),
            **(y_linear_kwargs or {}),
        }
        # noinspection PyArgumentList
        self.plots_["y_linear"] = ax.step(self.bins, self.values, **y_linear_kwargs)
        ax.set_xlim(-1, 257)
        ax.set_xticks(np.linspace(0, 256, 256 // 8 + 1))  # multiples of 8
        ax.set_xlabel("Voxel gray value [0, 255]")

        ax.set_ylim((0, 1.05 * max(self.values)))
        ax.set_ylabel("Proportion of voxels", color=y_linear_kwargs["color"], fontsize='large')
        ax.set_yticks([])

        y_log_kwargs = {
            **dict(linewidth=.75, color='red', label='log',),
            **(y_log_kwargs or {}),
        }
        # noinspection PyArgumentList
        self.plots_["y_log"] = axlog.step(self.bins, self.values, **y_log_kwargs)
        axlog.set_yscale("log")
        axlog.grid(axis='y', which='major', ls='--', color=y_log_kwargs["color"], alpha=.5)
        axlog.set_yticklabels(
            [f"10^{int(np.log10(t)):d}" for t in axlog.get_yticks()],
            c=y_log_kwargs["color"]
        )
        axlog.set_ylabel(
            "Proportion of voxels (log scale)",
            color=y_log_kwargs["color"],
            fontsize='large',
            rotation=-90,
            rotation_mode="anchor"
        )

        ax.set_title(f"Volume data histogram\nvolume={self.volume_name}")

        return self


@dataclass
class VoxelValueHistogramPerClassDisplay(Display):

    volume_name: str
    bins: list
    values_per_label: List[list]
    values_per_label_global_proportion: List[list]
    labels_idx: List[int]
    labels_names: List[str]

    ax_per_label_: Axes = field(init=False)
    ax_global_: Axes = field(init=False)

    def __post_init__(self):
        assert len(self.bins) == 256, f"{len(self.bins)=}"
        assert min(self.bins) == 0, f"{min(self.bins)=}"
        assert max(self.bins) == 255, f"{max(self.bins)}"

        for idx in self.labels_idx:
            assert (values_len := len(self.values_per_label[idx])) == 256, f"{values_len=} {idx=}"
            assert (values_len := len(self.values_per_label_global_proportion[idx])) == 256, f"{values_len=} {idx=}"

        # i want to get the vertical borders to show up
        self.bins = copy.copy(self.bins) + [256, 257]
        self.values_per_label = [
            [0] + copy.copy(self.values_per_label[idx]) + [0]
            for idx in self.labels_idx
        ]
        self.values_per_label_global_proportion = [
            [0] + copy.copy(self.values_per_label_global_proportion[idx]) + [0]
            for idx in self.labels_idx
        ]

    @property
    def title(self) -> str:
        return f"{self.volume_name}.data-histogram-per-label"

    def plot(self, axs: ndarray) -> "VoxelValueHistogramPerClassDisplay":
        assert (axs_shape := axs.shape) == (2,), f"{axs_shape=}"

        self.ax_per_label_ = ax_per_label = axs[0]
        self.ax_global_ = ax_global = axs[1]
        self.fig_ = fig = ax_per_label.figure

        assert isinstance(ax_per_label, Axes), f"{ax_per_label=}"
        assert isinstance(ax_global, Axes), f"{ax_global=}"

        for label_idx, label_name, label_hist, label_hist_global in zip(
            self.labels_idx, self.labels_names, self.values_per_label, self.values_per_label_global_proportion
        ):
            self.plots_[f"per_label.{label_idx=}"] = ax_per_label.step(
                self.bins,
                label_hist,
                linewidth=.75,
                label=f"{label_name} ({label_idx=})",
            )
            ax_per_label.set_xlim(0, 256)
            ax_per_label.set_xticks(np.linspace(0, 256, 256 // 8 + 1))
            ax_per_label.set_xlabel("Voxel gray value [0, 255]")

            self.plots_[f"global.{label_idx=}"] = ax_global.step(
                self.bins,
                label_hist_global,
                linewidth=.75,
                label=f"{label_name} ({label_idx=})",
            )
            ax_global.set_xlim(0, 256)
            ax_global.set_xticks(np.linspace(0, 256, 256 // 8 + 1))
            ax_global.set_xlabel("Voxel gray value [0, 255]")

        ax_per_label.set_ybound(lower=0)
        ax_per_label.set_ylabel("Proportion of voxels *per class*", fontsize='large')
        ax_per_label.legend()
        ax_per_label.set_title("Per class proportion\neach histogram is the proportion out of those of the same label")

        ax_global.set_yscale('log')
        ax_global.set_ybound(lower=0)
        ax_global.set_ylabel("Proportion of voxels *overall* (global) [log]", fontsize='large')
        ax_global.legend()
        ax_global.set_title("Global proportion\neach histogram is the proportion out of all voxels")
        ax_global.grid(axis='y', which='major', ls='--', alpha=.5)

        fig.suptitle(f"Volume data histogram per class\nvolume={self.volume_name}")

        return self
