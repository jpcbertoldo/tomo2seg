"""I just compute some values of reference for the losses."""
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
from numpy import ndarray
from tabulate import tabulate

from tomo2seg import losses as tomo2seg_losses
from tomo2seg.logger import logger


@dataclass
class TheoreticalModel(ABC):
    name: str

    @property
    def fullname(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    @property
    @abstractmethod
    def jaccard2_raveled_coeff(self) -> float:
        pass

    @property
    def jaccard2_raveled_loss(self) -> float:
        f"""
        :return: {tomo2seg_losses.jaccard2_flat.__module__}.{tomo2seg_losses.jaccard2_flat.__name__}
        """
        return 1 - self.jaccard2_raveled_coeff

    @property
    @abstractmethod
    def jaccard2_classwise_coeffs(self) -> List[float]:
        pass

    @property
    def jaccard2_classwise_losses(self) -> List[float]:
        f"""
        :return: {tomo2seg_losses.Jaccard2.__module__}.{tomo2seg_losses.Jaccard2.__name__}(i) for i in [0, ..., n_classes - 1] 
        """
        return [1 - coeff for coeff in self.jaccard2_classwise_coeffs]

    @property
    def jaccard2_macroavg_coeff(self) -> float:
        return sum(self.jaccard2_classwise_coeffs) / len(self.jaccard2_classwise_coeffs)

    @property
    def jaccard2_macroavg_loss(self) -> float:
        f"""
        :return: {tomo2seg_losses.jaccard2_macro_avg.__module__}.{tomo2seg_losses.jaccard2_macro_avg.__name__}
        """
        return 1 - self.jaccard2_macroavg_coeff


@dataclass
class UniformProbabilitiesClassifier(TheoreticalModel):
    f"""
    :attr proportions: the proportions of each class 
    """

    proportions: List[float]

    @property
    def n_classes(self) -> int:
        return len(self.proportions)

    @property
    def jaccard2_raveled_coeff(self) -> float:
        # n = number of samples
        # intersection = proba  # * n
        # true = 1  # * n
        # pred = n_classes * proba**2  # * n  = proba
        # jaccard2 = intersection / (true + pred - intersection)
        # jaccard2 = proba / (1 + proba - proba) = proba = 1/n_classes
        return 1. / self.n_classes

    @property
    def jaccard2_classwise_coeffs(self) -> List[float]:
        proportions = self.proportions
        for idx, p in enumerate(proportions):
            assert 0 < p < 1, f"{idx=} {p=}"
        assert (should_be_1 := sum(proportions)) == 1, f"{should_be_1=}"

        return [1. / ((self.n_classes ** 2) + (1 / pi) - 1) for pi in proportions]


@dataclass
class Order0Classifier(TheoreticalModel):
    """
    A classifier that only looks at the labels proportion and classifies everything as the majority class with 100%.
    The majority class is always supposed to be the index 0.
    :attr p0: the proportion of the majority class
    """

    n_classes: int
    p0: float

    @property
    def jaccard2_raveled_coeff(self) -> float:
        p0 = self.p0
        assert 0 < p0 < 1, f"{p0=}"
        # n = number of samples
        # intersection = p0  # * n
        # true = 1  # * n
        # pred = 1  # * n
        # jaccard2 = intersection / (true + pred - intersection) = p0 / (2 - p0)
        jaccard2 = p0 / (2 - p0)
        return jaccard2

    @property
    def jaccard2_classwise_coeffs(self) -> List[float]:
        p0 = self.p0
        assert 0 < p0 < 1, f"{p0=}"
        return [p0] + (self.n_classes - 1) * [0]


@dataclass
class ValueHistogramBased(TheoreticalModel):

    value_histograms: ndarray
    # class_proportions: ndarray = field(init=False)
    # global_histogram: ndarray = field(init=False)
    # decisions_map: ndarray = field(init=False)

    def __post_init__(self):
        raise NotImplementedError("I have to finish implementing this model.")

        assert (shape_1 := self.value_histograms.shape[1]) == 256, f"{shape_1=}"

        class_proportions = self.value_histograms.sum(axis=1)
        class_proportions /= class_proportions.sum()

        global_histogram = self.value_histograms.sum(axis=0)
        decisions_map = self.value_histograms.argmax(axis=0)

        accuracy_per_value = np.array([
            self.value_histograms[decisions_map[val], val] / global_histogram[val]
            for val in range(256)
        ])

        tp_per_class_per_value = np.array([
            [
                0. if decisions_map[val] != class_idx else
                accuracy_per_value[val]
                for val in range(256)
            ]
            for class_idx in range(self.value_histograms.shape[0])
        ])

        tp_per_class = np.average(tp_per_class_per_value, weights=global_histogram/global_histogram.sum())

    @property
    def jaccard2_raveled_coeff(self) -> float:

        pass

    @property
    def jaccard2_classwise_coeffs(self) -> List[float]:
        pass


# ======================================================= models =======================================================


# todo update these with the correct values
pa66gf30_proportions = [
    .809861,  # matrix
    .189801,  # fiber
    .000338,  # porosity
]

pa66gf30_uniform_proba = UniformProbabilitiesClassifier(
    name="pa66gf30",
    proportions=pa66gf30_proportions
)

pa66gf30_order0 = Order0Classifier(
    name="pa66gf30",
    n_classes=len(pa66gf30_proportions),
    p0=pa66gf30_proportions[0],
)

pa66gf30_models = [
    pa66gf30_uniform_proba,
    pa66gf30_order0,
]

models = []
models.extend(pa66gf30_models)

# ======================================================= losses =======================================================

global_losses = [
    attr
    for attr, _ in inspect.getmembers(TheoreticalModel)
    if isinstance(attr, str) and attr.endswith("loss")
]

classwise_losses = [
    attr
    for attr, _ in inspect.getmembers(TheoreticalModel)
    if isinstance(attr, str) and attr.endswith("classwise_losses")
]

losses = global_losses + classwise_losses

logger.debug(f"{global_losses=} {classwise_losses=}")
logger.info(f"{losses=}")

# ======================================================= table ========================================================


def get_loss(model: TheoreticalModel, attr: str, is_classwise: bool) -> str:

    try:
        loss = getattr(model, attr)

    except AttributeError:
        return "not def"

    if not is_classwise:
        return f"{loss :.3g}"

    else:
        return ", ".join(f'{float(f"{val:.2g}"):.2f}' for val in loss)


table = pd.DataFrame(
    data={
        **{
            attr: [getattr(model, attr) for model in models]
            for attr in ["fullname"]
        },
        **{
            attr.split("_loss")[0]: [get_loss(model, attr, False) for model in models]
            for attr in global_losses
        },
        **{
            attr.split("_losses")[0]: [get_loss(model, attr, True) for model in models]
            for attr in classwise_losses
        },
    }
).set_index("fullname")


if __name__ == "__main__":
    print(tabulate(table, headers="keys", tablefmt="psql"))
