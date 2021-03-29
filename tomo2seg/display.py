"""
Displays.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Union

import numpy as np
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.pyplot import Axes, Figure, cm
from numpy import ndarray

from .logger import logger


@dataclass
class Display(ABC):

    fig_: Figure = field(init=False)
    axs_: Dict[str, Union[Axes, ndarray]] = field(init=False, default_factory=dict)
    plots_: Dict[str, Union[List[Line2D], AxesImage]] = field(
        init=False, default_factory=dict
    )

    creation_time: int = field(init=False, default_factory=lambda: int(time.time()))

    @abstractmethod
    def plot(self, *args, **kwargs) -> "Display":
        pass

    @property
    def basename(self) -> str:
        return self.__class__.__name__.lower()


@dataclass
class SliceDisplay(Display):

    volume: str
    axis: str
    coord: int
    data: ndarray

    @property
    def name(self) -> str:
        return f"{self.volume}.{self.basename}.axis={self.axis}.coord={self.coord}"

    def plot(self, ax: Axes, vmin, vmax, imshow_kwargs: dict = None) -> "SliceDisplay":

        self.fig_ = ax.figure
        self.axs_["imshow"] = ax

        # overwrite default kwargs with given ones
        imshow_kwargs = {
            **dict(cmap=cm.gray, interpolation=None,),
            **(imshow_kwargs or dict()),
            **dict(vmin=vmin, vmax=vmax,),
        }
        self.plots_["imshow"] = ax.imshow(self.data, **imshow_kwargs)

        ax.set_title(f"{self.name}")
        ax.axis(False)

        self.post_imshow_hook()

        return self

    def post_imshow_hook(self):
        vmin, vmax = self.plots_["imshow"].get_clim()
        title = self.axs_["imshow"].get_title()
        title += f" {vmin=}, {vmax=}"
        self.axs_["imshow"].set_title(title)
