from matplotlib import pyplot as plt
from matplotlib.pyplot import Figure, Axes, cm
import numpy as np
from numpy import ndarray

from .logger import logger


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

