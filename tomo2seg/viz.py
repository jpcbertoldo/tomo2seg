from matplotlib import pyplot as plt
from matplotlib.pyplot import Figure, Axes, cm
import numpy as np
from numpy import ndarray


def tight_subplots(n: int, m: int, w: float, h: float) -> (Figure, Axes):
    fig, axs = plt.subplots(n, m, figsize=(w, h), sharex='all', sharey='all')
    fig.set_tight_layout(True)
    return fig, axs


def plot_orthogonal_slices(axs: ndarray, volume: ndarray, labels_mask: ndarray = None):

    if labels_mask is not None:
        assert volume.shape == labels_mask.shape

    xy_axis, yz_axis, xz_axis = axs[0, 0], axs[0, 1], axs[1, 0]
    axs[1, 1].axis(False)

    (height, width, depth) = volume.shape
    xy_z_coord, yz_x_coord, xz_y_coord = int(depth // 2), int(width // 2), int(height // 2)

    xy_axis.imshow(volume[:, :, xy_z_coord], cmap=cm.gray, interpolation=None)
    yz_axis.imshow(volume[yz_x_coord, :, :], cmap=cm.gray, interpolation=None)
    xz_axis.imshow(volume[:, xz_y_coord, :], cmap=cm.gray, interpolation=None)

    volume_masked = np.ma.masked_where(labels_mask, volume)
    xy_axis.imshow(volume_masked[:, :, xy_z_coord], cmap=cm.inferno, interpolation=None, alpha=0.5)
    yz_axis.imshow(volume_masked[yz_x_coord, :, :], cmap=cm.inferno, interpolation=None, alpha=0.5)
    xz_axis.imshow(volume_masked[:, xz_y_coord, :], cmap=cm.inferno, interpolation=None, alpha=0.5)

    xy_axis.set_title(f"XY :: z={xy_z_coord}")
    yz_axis.set_title(f"YZ :: x={yz_x_coord}")
    xz_axis.set_title(f"XZ :: y={xz_y_coord}")
    for ax in axs.ravel():
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    # todo generalize this so that I can plot slices from any coordinates --> pick randomly in usage


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

