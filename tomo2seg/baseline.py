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

    # =========================== jaccard2 ===========================

    @property
    @abstractmethod
    def jaccard2_raveled_coeff(self) -> float:
        pass

    @property
    @abstractmethod
    def jaccard2_classwise_coeffs(self) -> List[float]:
        pass

    @property
    def jaccard2_raveled_loss(self) -> float:
        f"""
        :return: {tomo2seg_losses.jaccard2_flat.__module__}.{tomo2seg_losses.jaccard2_flat.__name__}
        """

        if self.jaccard2_raveled_coeff is None:
            return None

        return 1 - self.jaccard2_raveled_coeff

    @property
    def jaccard2_classwise_losses(self) -> List[float]:
        f"""
        :return: {tomo2seg_losses.Jaccard2.__module__}.{tomo2seg_losses.Jaccard2.__name__}(i) for i in [0, ..., n_classes - 1]
        """

        if self.jaccard2_classwise_coeffs is None:
            return None

        return [1 - coeff for coeff in self.jaccard2_classwise_coeffs]

    @property
    def jaccard2_macroavg_coeff(self) -> float:

        if self.jaccard2_classwise_coeffs is None:
            return None

        return sum(self.jaccard2_classwise_coeffs) / len(self.jaccard2_classwise_coeffs)

    @property
    def jaccard2_macroavg_loss(self) -> float:
        f"""
        :return: {tomo2seg_losses.jaccard2_macro_avg.__module__}.{tomo2seg_losses.jaccard2_macro_avg.__name__}
        """

        if self.jaccard2_macroavg_coeff is None:
            return None

        return 1 - self.jaccard2_macroavg_coeff

    # =========================== jaccard ===========================

    @property
    @abstractmethod
    def jaccard_raveled_coeff(self) -> float:
        pass

    @property
    @abstractmethod
    def jaccard_classwise_coeffs(self) -> List[float]:
        pass

    @property
    def jaccard_macroavg_coeff(self) -> float:

        if self.jaccard_classwise_coeffs is None:
            return None

        return sum(self.jaccard_classwise_coeffs) / len(self.jaccard_classwise_coeffs)

    @property
    def jaccard_macroavg_loss(self) -> float:

        if self.jaccard_macroavg_coeff is None:
            return None

        return 1 - self.jaccard_macroavg_coeff


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
        return 1.0 / self.n_classes

    @property
    def jaccard2_classwise_coeffs(self) -> List[float]:
        proportions = self.proportions
        for idx, p in enumerate(proportions):
            assert 0 < p < 1, f"{idx=} {p=}"
        assert (should_be_1 := sum(proportions)) == 1, f"{should_be_1=}"

        return [1.0 / ((self.n_classes ** 2) + (1 / pi) - 1) for pi in proportions]

    @property
    def jaccard_raveled_coeff(self) -> float:
        return None  # undef because it never takes a decision

    @property
    def jaccard_classwise_coeffs(self) -> List[float]:
        return None  # undef because it never takes a decision


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

    @property
    def jaccard_raveled_coeff(self) -> float:
        # it is the same because the square in the denominator always gives 1
        return self.jaccard2_raveled_coeff

    @property
    def jaccard_classwise_coeffs(self) -> List[float]:
        # it is the same because the square in the denominator always gives 1
        return self.jaccard2_classwise_coeffs


@dataclass
class BinwiseOrder0Classifier(TheoreticalModel):

    classwise_histograms: ndarray  # counts

    n_classes_: int = field(init=False)
    decisions_map_: ndarray = field(init=False)

    def __post_init__(self):

        self.n_classes_ = self.classwise_histograms.shape[0]

        assert (shape_1 := self.classwise_histograms.shape[1]) == 256, f"{shape_1=}"

        self.class_proportions_ = self.classwise_histograms.sum(axis=1)
        self.class_proportions_ = (
            self.class_proportions_ / self.class_proportions_.sum()
        )

        self.binwise_class_proportions_ = (
            self.classwise_histograms
            / self.classwise_histograms.sum(axis=0, keepdims=True)
        )

        self.binwise_class_proportions_ = (
            self.binwise_class_proportions_
            / self.binwise_class_proportions_.sum(axis=0, keepdims=True)
        )

        assert np.all(np.isclose(self.binwise_class_proportions_.sum(axis=0), 1))

        self.decisions_map_ = self.binwise_class_proportions_.argmax(axis=0)

        self.norm_histogram_ = self.classwise_histograms.sum(axis=0)
        self.norm_histogram_ = self.norm_histogram_ / self.norm_histogram_.sum(axis=0)
        self.norm_histogram_ = self.norm_histogram_ / self.norm_histogram_.sum(axis=0)

        self.normalized_classwise_histograms_ = (
            self.classwise_histograms / self.classwise_histograms.sum()
        )

    @property
    def jaccard2_raveled_coeff(self) -> float:
        intersec = np.sum(
            [
                self.normalized_classwise_histograms_[self.decisions_map_[val], val]
                for val in range(256)
            ]
        )
        return intersec / (2 - intersec)

    @property
    def jaccard2_classwise_coeffs(self) -> List[float]:

        coeffs = []

        for k in range(self.n_classes_):

            intersec = np.sum(
                [
                    self.normalized_classwise_histograms_[k, val]
                    if self.decisions_map_[val] == k
                    else 0
                    for val in range(256)
                ]
            )

            gt = self.class_proportions_[k]

            pred = np.sum(
                [
                    self.norm_histogram_[val] if self.decisions_map_[val] == k else 0
                    for val in range(256)
                ]
            )

            coeffs.append(intersec / (gt + pred - intersec))

        return coeffs

    @property
    def jaccard_raveled_coeff(self) -> float:
        # it is the same because the square in the denominator always gives 1
        return self.jaccard2_raveled_coeff

    @property
    def jaccard_classwise_coeffs(self) -> List[float]:
        # it is the same because the square in the denominator always gives 1
        return self.jaccard2_classwise_coeffs


def main():

    # ===================================================== models =====================================================

    # todo update these with the correct values
    pa66gf30_proportions = [
        0.809861,  # matrix
        0.189801,  # fiber
        0.000338,  # porosity
    ]

    pa66gf30_classwise_histograms = np.load(
        "../data/PA66GF30.v1/ground-truth-analysis/histogram-per-label.npy"
    )

    pa66gf30_models = [
        UniformProbabilitiesClassifier(
            name="pa66gf30", proportions=pa66gf30_proportions
        ),
        Order0Classifier(
            name="pa66gf30",
            n_classes=len(pa66gf30_proportions),
            p0=pa66gf30_proportions[0],
        ),
        BinwiseOrder0Classifier(
            name="pa66gf30", classwise_histograms=pa66gf30_classwise_histograms,
        ),
    ]

    models = []
    models.extend(pa66gf30_models)

    # ===================================================== losses =====================================================

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

    # ===================================================== coeffs =====================================================

    global_coeffs = [
        attr
        for attr, _ in inspect.getmembers(TheoreticalModel)
        if isinstance(attr, str) and attr.endswith("coeff")
    ]

    classwise_coeffs = [
        attr
        for attr, _ in inspect.getmembers(TheoreticalModel)
        if isinstance(attr, str) and attr.endswith("classwise_coeffs")
    ]

    coeffs = global_coeffs + classwise_coeffs

    logger.debug(f"{global_coeffs=} {classwise_coeffs=}")
    logger.info(f"{coeffs=}")

    # ===================================================== table ======================================================

    def get_value(model: TheoreticalModel, attr: str, is_classwise: bool) -> str:

        try:
            val = getattr(model, attr)

        except AttributeError:
            return "not def"

        if val is None:
            return "not def"

        if not is_classwise:
            return f'{float(f"{val:.4g}"):.2%}'

        else:
            return ", ".join(f'{float(f"{v:.4g}"):.2%}' for v in val)

    table_losses = pd.DataFrame(
        data={
            **{
                attr: [getattr(model, attr) for model in models]
                for attr in ["fullname"]
            },
            **{
                attr.split("_loss")[0]: [
                    get_value(model, attr, False) for model in models
                ]
                for attr in global_losses
            },
            **{
                attr.split("_losses")[0]: [
                    get_value(model, attr, True) for model in models
                ]
                for attr in classwise_losses
            },
        }
    ).set_index("fullname")

    table_coeffs = pd.DataFrame(
        data={
            **{
                attr: [getattr(model, attr) for model in models]
                for attr in ["fullname"]
            },
            **{
                attr.split("_coeff")[0]: [
                    get_value(model, attr, False) for model in models
                ]
                for attr in global_coeffs
            },
            **{
                attr.split("_coeffs")[0]: [
                    get_value(model, attr, True) for model in models
                ]
                for attr in classwise_coeffs
            },
        }
    ).set_index("fullname")

    print("losses:")
    print(tabulate(table_losses, headers="keys", tablefmt="psql"))

    print("coefficients:")
    print(tabulate(table_coeffs, headers="keys", tablefmt="psql"))


if __name__ == "__main__":

    main()
