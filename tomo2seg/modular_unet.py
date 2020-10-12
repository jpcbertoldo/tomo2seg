"""Modular U-Net"""

# Installed packages
import tensorflow
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    concatenate,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    GaussianNoise,
    Dropout,
    Conv2DTranspose,
    SeparableConv2D,
    Activation,
    add,
    BatchNormalization,
)


# todo use model class instead
def generic_unet_block(name, nb_filters_1, nb_filters_2, kernel_size, res, batch_norm, dropout, separable_conv=True):

    conv_layer = SeparableConv2D if separable_conv else Conv2D

    def fn(tensor):

        x = tensor

        if dropout > 0:
            x = Dropout(dropout)(x)

        skip = x

        x = conv_layer(nb_filters_1, kernel_size, padding="same", name=f"{name}-conv1")(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = Activation("relu")(x)

        if dropout > 0:
            x = Dropout(dropout)(x)

        x = conv_layer(nb_filters_2, kernel_size, padding="same", name=f"{name}-conv2")(x)
        if batch_norm:
            x = BatchNormalization()(x)

        if res:
            skip = conv_layer(nb_filters_2, kernel_size, padding="same", name=f"{name}-conv-skip")(skip)
            x = add([x, skip])

        x = Activation("relu")(x)

        return x

    return fn


def generic_unet_down(conv_sampling, nb_filters=None, name=None):
    if conv_sampling:

        if nb_filters is None:
            raise ValueError("When conv_sampling is True, nb_filters should be given")

        def fn(tensor):
            x = Conv2D(nb_filters, 3, padding="same", strides=(2, 2), name=f"{name}-DOWN")(tensor)
            return x

    else:

        def fn(tensor):
            x = MaxPooling2D(pool_size=(2, 2))(tensor)
            return x

    return fn


def generic_unet_up(conv_sampling, nb_filters=None, name=None):
    if conv_sampling:

        if nb_filters is None:
            raise ValueError("When conv_sampling is True, nb_filters should be given")

        def fn(tensor):
            x = Conv2DTranspose(nb_filters, 3, padding="same", strides=(2, 2), name=f"{name}-UP")(tensor)
            return x

    else:

        def fn(tensor):
            x = UpSampling2D(size=(2, 2))(tensor)
            return x

    return fn


def u_net(
    shape,
    nb_filters_0,
    output_channels,
    name=None,
):
    """Modular U-Net.

    Note that the dimensions of the input images should be
    multiples of 16.
"""
    import functools
    unet_block = functools.partial(
        generic_unet_block,
        kernel_size=3,
        res=True,
        batch_norm=True,
        dropout=0.03,
        separable_conv=True,
    )
    unet_down = functools.partial(
        generic_unet_down,
        conv_sampling=True,
    )

    unet_up = functools.partial(
        generic_unet_up,
        conv_sampling=True,
    )

    depth = 4
    # sigma_noise = 0.03

    x = x0 = Input(shape)

    skips = {}
    for i in range(depth):
        nb_filters_begin = nb_filters_0 * 2 ** i
        nb_filters_end = nb_filters_0 * 2 ** (i + 1)
        x = unet_block(f"encoder-block-{i}", nb_filters_begin, nb_filters_end)(x)
        skips[i] = x
        x = unet_down(nb_filters=nb_filters_end, name=f"encoder-block-{i}")(x)

    nb_filters_begin = nb_filters_0 * 2 ** depth
    nb_filters_end = nb_filters_0 * 2 ** (depth + 1)
    x = unet_block(f"encoder-block-{depth}", nb_filters_begin, nb_filters_end)(x)

    for i in reversed(range(depth)):
        nb_filters_up = nb_filters_0 * 2 ** (i + 2)
        nb_filters_conv = nb_filters_0 * 2 ** (i + 1)
        x = unet_up(nb_filters=nb_filters_up, name=f"decoder-block-{i}")(x)
        x_skip = skips[i]
        x = concatenate([x_skip, x], axis=3)
        x = unet_block(f"decoder-block-{i}", nb_filters_conv, nb_filters_conv)(x)

    """
    nb_filters_1 = nb_filters_0 * 2
    x1 = unet_block("conv1", nb_filters_0, nb_filters_1)(x0)

    x2 = unet_down(nb_filters_1)(x1)

    nb_filters_2 = nb_filters_0 * 4
    x2 = unet_block("conv2", nb_filters_1, nb_filters_2)(x2)

    x3 = unet_down(nb_filters_2)(x2)

    nb_filters_3 = nb_filters_0 * 8
    x3 = unet_block("conv3", nb_filters_2, nb_filters_3)(x3)

    x4 = unet_down(nb_filters_3)(x3)

    nb_filters_4 = nb_filters_0 * 16
    x4 = unet_block("conv4", nb_filters_3, nb_filters_4)(x4)

    x5 = unet_down(nb_filters_4)(x4)

    nb_filters_5 = nb_filters_0 * 32
    x5 = unet_block("conv5", nb_filters_4, nb_filters_5)(x5)
    x6 = unet_up(nb_filters_5)(x5)
    x6 = concatenate([x6, x4], axis=3)
    x6 = unet_block("conv6", nb_filters_4, nb_filters_4)(x6)

    x7 = unet_up(nb_filters_4)(x6)
    x7 = concatenate([x7, x3], axis=3)
    x7 = unet_block("conv7", nb_filters_3, nb_filters_3)(x7)

    x8 = unet_up(nb_filters_3)(x7)
    x8 = concatenate([x8, x2], axis=3)
    x8 = unet_block("conv8", nb_filters_2, nb_filters_2)(x8)

    x9 = unet_up(nb_filters_2)(x8)
    x9 = concatenate([x9, x1], axis=3)
    x9 = unet_block("conv9", nb_filters_1, nb_filters_1)(x9)"""

    # x = GaussianNoise(sigma_noise)(x)

    x = Conv2D(output_channels, 1, activation="softmax", name="out")(x)

    return Model(x0, x, name=name)
