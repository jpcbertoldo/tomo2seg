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

    @classmethod
    def get_losses_names(cls):

        global_losses = [
            attr
            for attr, _ in inspect.getmembers(cls)
            if isinstance(attr, str) and attr.endswith("loss")
        ]

        classwise_losses = [
            attr
            for attr, _ in inspect.getmembers(cls)
            if isinstance(attr, str) and attr.endswith("classwise_losses")
        ]

        return global_losses, classwise_losses


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
class MulticlassPerfectRecall(TheoreticalModel):
    """
    A classifier that can perfectly distinguish a subset of the classes. 
    All the rest is confounded with one of the learned classes.
    Therefore, some classes will have a perfect recall, while others will have
    a 0 recall, classifying all of it as another (user-selected) class.
   
    :attr proportions: the proportions of each class 
    :attr binary_confusion_matrix: the class i is classified as class j if binary_confusion_matrix[i][j] == 1
    """

    proportions: List[float]
    binary_confusion_matrix: List[List[int]]

    def __post_init__(self):

        assert all(v == 0 or v == 1 for row in self.binary_confusion_matrix for v in row)

        n_classes = len(self.proportions)

        assert len(self.binary_confusion_matrix) == n_classes

        for idx, row in enumerate(self.binary_confusion_matrix):
            assert len(row) == n_classes, f"{idx}"

        assert any(self.binary_confusion_matrix[idx][idx] == 1 for idx in range(n_classes))

        for prop in self.proportions:

            assert 0 < prop < 1, f"{prop}"

        assert sum(self.proportions) == 1

    @property
    def jaccard2_raveled_coeff(self) -> float:
        n_classes = len(self.proportions)
        # n = number of samples
        intersec = sum(
            self.proportions[idx] 
            for idx in range(n_classes)
            if self.binary_confusion_matrix[idx][idx] == 1
        )  # * n
        # true = 1  # * n
        # pred = 1  # * n
        jaccard2 = intersec / (2 - intersec)
        return jaccard2

    @property
    def jaccard2_classwise_coeffs(self) -> List[float]:
        coeffs = []
        n_classes = len(self.proportions)
        for idx in range(n_classes):
            intersection = self.proportions[idx] * self.binary_confusion_matrix[idx][idx]
            true = self.proportions[idx]
            pred = sum(
                self.proportions[idx] * self.binary_confusion_matrix[row_idx][idx] 
                for row_idx in range(n_classes)
            )
            coeffs.append(intersection / (true + pred - intersection))
        return coeffs 


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


# ================================================ losses names / models ================================================

global_losses, classwise_losses = TheoreticalModel.get_losses_names()
losses = global_losses + classwise_losses
models = []

logger.debug(f"{global_losses=} {classwise_losses=}")
logger.info(f"{losses=}")

# ======================================================= PA66GF30 =======================================================

# ============================= class distribution =============================

# todo update these with the precise values
pa66gf30_proportions = [
    .809861,  # matrix
    .189801,  # fiber
    .000338,  # porosity
]

# =================================== models ===================================

pa66gf30_models = [
    UniformProbabilitiesClassifier(
        name="pa66gf30",
        proportions=pa66gf30_proportions,
    ),
    Order0Classifier(
        name="pa66gf30",
        n_classes=len(pa66gf30_proportions),
        p0=pa66gf30_proportions[0],
    ),
    MulticlassPerfectRecall(
        name="pa66gf30:matrix,fiber:porosity->matrix",
        proportions=pa66gf30_proportions,
        binary_confusion_matrix=[
            [1, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
        ],
    )
]

models.extend(pa66gf30_models)

# ====================================================== fracture00 ======================================================

# ============================= class distribution =============================

fracture00_proportions = [
    .260358,  # exterior
    .713693,  # inside
    .021588,  # defect
    .000163,  # porosity
    .004198,  # crack
]

# =================================== models ===================================

fracture00_models = [
    UniformProbabilitiesClassifier(
        name="fracture00",
        proportions=fracture00_proportions
    ),
    Order0Classifier(
        name="fracture00",
        n_classes=len(fracture00_proportions),
        p0=fracture00_proportions[1],
    ),
    MulticlassPerfectRecall(
        name="fracture00:exterior,inside:others->inside",
        proportions=fracture00_proportions,
        binary_confusion_matrix=[
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
        ],
    ),
    MulticlassPerfectRecall(
        name="fracture00:exterior,inside,defect:others->inside",
        proportions=fracture00_proportions,
        binary_confusion_matrix=[
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
        ],
    ),
]

models.extend(fracture00_models)


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
            attr: [get_loss(model, attr, False) for model in models]
            for attr in global_losses
        },
        **{
            attr: [get_loss(model, attr, True) for model in models]
            for attr in classwise_losses
        },
    }
).set_index("fullname")


if __name__ == "__main__":
    print(tabulate(table, headers="keys", tablefmt="psql"))
