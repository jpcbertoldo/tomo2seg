"""Things to be used with keras.callbacks.LearningRateScheduler"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

import numpy as np

from .logger import logger


def log_schedule_factory(start_pow10, stop_pow10, n_per_scale, wait, offset_epoch=0):
    """From 10 ** start_pow10 until 10 ** stop_pow10 with n_per_scale points between each scale of 10."""
    n = (n_per_scale + 1) * abs(stop_pow10 - start_pow10) + 1
    schedule = np.array(wait * [10 ** start_pow10])
    schedule = np.concatenate([schedule, np.logspace(start_pow10, stop_pow10, n)])
    logger.info(f"log schedule {n=} {wait=} {wait+n=}")

    def log_schedule(epoch):
        epoch -= offset_epoch
        if epoch >= schedule.shape[0]:
            return schedule[-1]
        return schedule[epoch]

    return log_schedule


@dataclass
class Schedule(ABC):

    offset_epoch: int

    @property
    @abstractmethod
    def n(self) -> int:
        """number of epochs scheduled"""
        pass

    @property
    def range(self) -> Tuple[int, int]:
        """epoch start/end"""
        return self.offset_epoch, self.offset_epoch + self.n

    @abstractmethod
    def __call__(self, epoch) -> float:
        pass


@dataclass
class LogSpaceSchedule(Schedule):
    """
    Start and stop are powers of 10.
    From 10 ** start until 10 ** stop with n_between_scales points between each scale of 10.
    When the epoch is above the number of epochs expected the last value is returned.
    """

    wait: int
    start: int
    stop: int
    n_between_scales: int

    schedule_: List[float] = field(init=False)

    def __post_init__(self):
        n = (self.n_between_scales + 1) * abs(self.stop - self.start) + 1

        schedule = np.concatenate([
            np.array(self.wait * [10 ** self.start]), 
            np.logspace(self.start, self.stop, n)
        ]).tolist()
        self.schedule_ = schedule
        logger.info(f"{self.__class__.__name__} ==> {self.n=}")

    @property
    def n(self) -> int:
        return len(self.schedule_)
    
    def __call__(self, epoch) -> float:
        assert epoch >= self.offset_epoch, f"{epoch=} {self.offset_epoch=}"
        epoch -= self.offset_epoch
        return self.schedule_[epoch] if epoch < len(self.schedule_) else self.schedule_[-1]


@dataclass
class LinSpaceSchedule(Schedule):
    """
    """

    wait: int
    start: float
    stop: float
    n_between: int

    schedule_: List[float] = field(init=False)

    def __post_init__(self):
        self.schedule_ = np.concatenate([
            np.array(self.wait * [self.start]), 
            np.linspace(self.start, self.stop, self.n_between + 2)
        ]).tolist()
        logger.info(f"{self.__class__.__name__} ==> {self.n=}")

    @property
    def n(self) -> int:
        return len(self.schedule_)
    
    def __call__(self, epoch) -> float:
        assert epoch >= self.offset_epoch, f"{epoch=} {self.offset_epoch=}"
        epoch -= self.offset_epoch
        return self.schedule_[epoch] if epoch < len(self.schedule_) else self.schedule_[-1]

    
@dataclass
class ComposedSchedule(Schedule):

    sub_schedules: List[Schedule]
    epoch_mapping_: Dict[int, int] = field(init=False)
    last_scheduled_epoch_: int = field(init=False)

    def __post_init__(self):

        self.epoch_mapping_ = {}

        for sched_idx, (sched, sched_after) in enumerate(zip(
                self.sub_schedules, self.sub_schedules[1:] + [None]  # none = "after the last schedule"
        )):

            if sched_after is not None:
                assert sched.range[1] == sched_after.range[0], f"{sched_idx=} {sched.range} {sched_after.range}"

            for epoch in range(*sched.range):
                self.epoch_mapping_[epoch] = sched_idx

            if sched_after is None:
                self.last_scheduled_epoch_ = epoch

        logger.info(f"{self.__class__.__name__} ==> {self.n=}")

    @property
    def n(self) -> int:
        return sum(sched.n for sched in self.sub_schedules)

    def __call__(self, epoch) -> float:
        assert epoch >= self.offset_epoch, f"{epoch=} {self.offset_epoch=}"

        if epoch > self.last_scheduled_epoch_:
            return self.sub_schedules[-1](epoch)

        return self.sub_schedules[self.epoch_mapping_[epoch]](epoch)
    

def get_schedule00():
    """
    teeth log lr schedule
    """
    return ComposedSchedule(
        offset_epoch=0,
        sub_schedules=[
            LogSpaceSchedule(0, wait=0, start=-4, stop=-3, n_between_scales=8), 
            LogSpaceSchedule(10, wait=20, start=-3, stop=-4, n_between_scales=8),
            LogSpaceSchedule(40, wait=0, start=-4, stop=-3, n_between_scales=18),
            LogSpaceSchedule(60, wait=20, start=-3, stop=-4, n_between_scales=18),
            LogSpaceSchedule(100, wait=0, start=-4, stop=-3, n_between_scales=18),
            LogSpaceSchedule(120, wait=20, start=-3, stop=-4, n_between_scales=18),
            LogSpaceSchedule(160, wait=50, start=-4, stop=-5, n_between_scales=48),
        ]
    )