from dataclasses import dataclass

from tensorflow.keras import backend as K
from tensorflow.python.keras.metrics import Metric

SMOOTH = 1.0


def jaccard2_flat(y_true, y_pred, smooth=SMOOTH):
    y_true_ndim = K.ndim(y_true)
    y_pred_ndim = K.ndim(y_pred)

    # 3d or 2d cases
    assert (y_true_ndim == 4 and y_pred_ndim == 5) or (
        y_true_ndim == 3 and y_pred_ndim == 4
    ), f"{y_true_ndim=} {y_pred_ndim=}"  # batch_idx, x, y, z (, class)

    # one hot encoding
    n_classes = y_pred.shape[-1]
    y_true = K.cast(K.one_hot(K.cast(y_true, "int32"), n_classes), y_pred.dtype)

    axis = None  # over every dimension
    intersection = K.sum(y_true * y_pred, axis=axis)
    # y_true is ohe, so y_true ** 2 is the same thing
    union = K.sum(y_true, axis=axis) + K.sum(y_pred ** 2, axis=axis) - intersection
    jaccards = (intersection + smooth) / (union + smooth)
    return 1.0 - K.mean(jaccards)


def jaccard2_macro_avg(y_true, y_pred, smooth=SMOOTH):
    y_true_ndim = K.ndim(y_true)
    y_pred_ndim = K.ndim(y_pred)

    # 3d or 2d cases
    assert (y_true_ndim == 4 and y_pred_ndim == 5) or (
        y_true_ndim == 3 and y_pred_ndim == 4
    ), f"{y_true_ndim=} {y_pred_ndim=}"  # batch_idx, x, y, z (, class)

    # one hot encoding
    n_classes = y_pred.shape[-1]
    y_true = K.cast(K.one_hot(K.cast(y_true, "int32"), n_classes), y_pred.dtype)

    axis = (0, 1, 2, 3) if y_pred_ndim == 5 else (0, 1, 2,)  # batch_idx, x, y(, z)
    intersection = K.sum(y_true * y_pred, axis=axis)
    # y_true is ohe, so y_true ** 2 is the same thing
    union = K.sum(y_true, axis=axis) + K.sum(y_pred ** 2, axis=axis) - intersection
    jaccards = (intersection + smooth) / (union + smooth)
    return 1.0 - K.mean(jaccards)


@dataclass(unsafe_hash=True)
class Jaccard2:
    class_idx: int

    def __post_init__(self):
        class_idx = self.class_idx
        self.__name__ = f"{self.__class__.__name__}.{class_idx=}".lower()

    def __call__(self, y_true, y_pred, smooth=SMOOTH):
        y_true_ndim = K.ndim(y_true)
        y_pred_ndim = K.ndim(y_pred)

        # 3d or 2d cases
        assert (y_true_ndim == 4 and y_pred_ndim == 5) or (
            y_true_ndim == 3 and y_pred_ndim == 4
        ), f"{y_true_ndim=} {y_pred_ndim=}"  # batch_idx, x, y, z (, class)

        y_true = K.cast(y_true == self.class_idx, y_pred.dtype)
        if y_pred_ndim == 4:
            y_pred = y_pred[:, :, :, self.class_idx]
        else:
            y_pred = y_pred[:, :, :, :, self.class_idx]

        intersection = K.sum(y_true * y_pred)
        # y_true is ohe, so y_true ** 2 is the same thing
        union = K.sum(y_true) + K.sum(y_pred ** 2) - intersection
        return 1.0 - ((intersection + smooth) / (union + smooth))
