"""Modular U-Net"""

import functools

# Installed packages
from enum import Enum

import tensorflow
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import layers

from tomo2seg.logger import logger, dict2str


class ConvLayer(Enum):
    conv2d = 0
    conv2d_separable = 1
    conv3d = 10
    conv3d_separable = 11


# todo use model class instead
def generic_unet_block(
    name, nb_filters_1, nb_filters_2, convlayer: ConvLayer, kernel_size, res, batch_norm, dropout, return_layers: bool = False,
):
    if convlayer == ConvLayer.conv2d:
        conv_layer = layers.Conv2D

    elif convlayer == ConvLayer.conv2d_separable:
        conv_layer = layers.SeparableConv2D

    elif convlayer == ConvLayer.conv3d:
        conv_layer = layers.Conv3D

    elif convlayer == ConvLayer.conv3d_separable:
        raise NotImplementedError(f"{convlayer.name=}")

    else:
        raise ValueError(f"{convlayer=}")

    def fn(tensor, reused_layers: dict = None,):
        
        layers_dic = {}

        x = tensor
        skip = x
        
        layer_name = f"{name}-conv1" if name is not None else None 
        layer = conv_layer(
            nb_filters_1, activation="linear", kernel_size=kernel_size, padding="same",
            name=layer_name
        )
        if reused_layers is not None:
            layer = reused_layers[layer_name]
        layers_dic[layer_name] = layer
        x = layer(x)

        if batch_norm:
            layer_name = f"{name}-conv1-bn" if name is not None else None 
            layer = layers.BatchNormalization(name=layer_name)
            if reused_layers is not None:
                layer = reused_layers[layer_name]
            layers_dic[layer_name] = layer
            x = layer(x)

        layer_name = f"{name}-conv1-relu" if name is not None else None 
        layer = layers.Activation("relu", name=layer_name)
        if reused_layers is not None:
            layer = reused_layers[layer_name]
        layers_dic[layer_name] = layer
        x = layer(x)

        if dropout > 0:
            layer_name = f"{name}-conv1-dropout" if name is not None else None 
            layer = layers.Dropout(dropout, name=layer_name)
            if reused_layers is not None:
                layer = reused_layers[layer_name]
            layers_dic[layer_name] = layer
            x = layer(x)

        layer_name = f"{name}-conv2" if name is not None else None 
        layer = conv_layer(
            nb_filters_2, activation="linear", kernel_size=kernel_size, padding="same",
            name=layer_name
        )
        if reused_layers is not None:
            layer = reused_layers[layer_name]
        layers_dic[layer_name] = layer
        x = layer(x)

        if batch_norm:
            layer_name = f"{name}-conv2-bn" if name is not None else None 
            layer = layers.BatchNormalization(name=layer_name)
            if reused_layers is not None:
                layer = reused_layers[layer_name]
            layers_dic[layer_name] = layer
            x = layer(x)

        if res:
            # todo check if this layer should have a batchnorm
            layer_name = f"{name}-conv-skip" if name is not None else None 
            layer = conv_layer(
                nb_filters_2, activation="linear", kernel_size=kernel_size, padding="same",
                name=layer_name
            )
            if reused_layers is not None:
                layer = reused_layers[layer_name]
            layers_dic[layer_name] = layer
            skip = layer(skip)

            if batch_norm:
                layer_name = f"{name}-conv-skip-bn" if name is not None else None 
                layer = layers.BatchNormalization(name=layer_name)
                if reused_layers is not None:
                    layer = reused_layers[layer_name]
                layers_dic[layer_name] = layer
                skip = layer(skip)

            layer_name = f"{name}-residual" if name is not None else None 
            layer = layers.Add(name=layer_name)
            if reused_layers is not None:
                layer = reused_layers[layer_name]
            layers_dic[layer_name] = layer
            x = layer([x, skip])

        layer_name = f"{name}-conv2-relu" if name is not None else None 
        layer = layers.Activation("relu", name=layer_name)
        if reused_layers is not None:
            layer = reused_layers[layer_name]
        layers_dic[layer_name] = layer
        x = layer(x)

        if dropout > 0:
            layer_name = f"{name}-conv2-dropout" if name is not None else None 
            layer = layers.Dropout(dropout, name=layer_name)
            if reused_layers is not None:
                layer = reused_layers[layer_name]
            layers_dic[layer_name] = layer
            x = layer(x)
            
        if return_layers:
            return x, layers_dic

        return x
    
    return fn


def generic_unet_down(
    conv_sampling, convlayer, nb_filters, name, batchnorm=False,
    return_layers: bool = False,
):
    
    if conv_sampling:

        if convlayer in (ConvLayer.conv2d, ConvLayer.conv2d_separable):
            conv_layer = layers.Conv2D

        elif convlayer in (ConvLayer.conv3d, ConvLayer.conv3d_separable):
            conv_layer = layers.Conv3D

        else:
            raise ValueError(f"{convlayer=}")

        if nb_filters is None:
            raise ValueError("When conv_sampling is True, nb_filters should be given")

        def fn(tensor, reused_layers: dict = None,):
            layers_dic = {}
            
            layer_name = f"{name}-DOWN" if name is not None else None
            layer = conv_layer(
                nb_filters, kernel_size=3, padding="same", strides=2, name=layer_name
            )
            if reused_layers is not None:
                layer = reused_layers[layer_name]
            layers_dic[layer_name] = layer
            x = layer(tensor)            
        
            if batchnorm:
                layer_name = f"{name}-DOWN-bn" if name is not None else None
                layer = layers.BatchNormalization(name=layer_name)
                if reused_layers is not None:
                    layer = reused_layers[layer_name]
                layers_dic[layer_name] = layer
                x = layer(x)
            
            layer_name = f"{name}-DOWN-relu" if name is not None else None
            layer = layers.Activation("relu", name=layer_name)
            if reused_layers is not None:
                layer = reused_layers[layer_name]
            layers_dic[layer_name] = layer
            x = layer(x)
            
            if return_layers:
                return x, layers_dic
            
            return x

    else:

        if convlayer in (ConvLayer.conv2d, ConvLayer.conv2d_separable):
            max_pooling = layers.MaxPooling2D
            pool_size = (2, 2)

        elif convlayer in (ConvLayer.conv3d, ConvLayer.conv3d_separable):
            max_pooling = layers.MaxPooling3D
            pool_size = (2, 2, 2)

        else:
            raise ValueError(f"{convlayer=}")

        def fn(tensor, reused_layers: dict = None,):
            layers_dic = {}
            
            layer_name = f"{name}-DOWN" if name is not None else None
            layer = max_pooling(pool_size=pool_size, name=layer_name)
            if reused_layers is not None:
                layer = reused_layers[layer_name]
            layers_dic[layer_name] = layer
            x = layer(tensor)    
            
            if return_layers:
                return x, layers_dic
            
            return x
    
    return fn


def generic_unet_up(
    conv_sampling, convlayer, nb_filters, name, batchnorm=False,
    return_layers: bool = False,
):
    if conv_sampling:

        if convlayer in (ConvLayer.conv2d, ConvLayer.conv2d_separable):
            conv_layer = layers.Conv2DTranspose

        elif convlayer in (ConvLayer.conv3d, ConvLayer.conv3d_separable):
            conv_layer = layers.Conv3DTranspose

        else:
            raise ValueError(f"{convlayer=}")

        if nb_filters is None:
            raise ValueError("When conv_sampling is True, nb_filters should be given")

        def fn(tensor, reused_layers: dict = None,):
            
            layers_dic = {}
            
            layer_name = f"{name}-UP" if name is not None else None
            layer = conv_layer(
                nb_filters, kernel_size=3, padding="same", strides=2, name=layer_name
            )
            if reused_layers is not None:
                layer = reused_layers[layer_name]
            layers_dic[layer_name] = layer
            x = layer(tensor)            
        
            if batchnorm:
                layer_name = f"{name}-UP-bn" if name is not None else None
                layer = layers.BatchNormalization(name=layer_name)
                if reused_layers is not None:
                    layer = reused_layers[layer_name]
                layers_dic[layer_name] = layer
                x = layer(x)
            
            layer_name = f"{name}-UP-relu" if name is not None else None
            layer = layers.Activation("relu", name=layer_name)
            if reused_layers is not None:
                layer = reused_layers[layer_name]
            layers_dic[layer_name] = layer
            x = layer(x)
            
            if return_layers:
                return x, layers_dic
            
            return x

    else:

        if convlayer in (ConvLayer.conv2d, ConvLayer.conv2d_separable):
            up_sampling = layers.UpSampling2D
            pool_size = (2, 2)

        elif convlayer in (ConvLayer.conv3d, ConvLayer.conv3d_separable):
            up_sampling = layers.UpSampling3D
            pool_size = (2, 2, 2)

        def fn(tensor, reused_layers: dict = None,):
            
            layers_dic = {}
            
            layer_name = f"{name}-UP" if name is not None else None
            layer = up_sampling(size=pool_size, name=layer_name)
            if reused_layers is not None:
                layer = reused_layers[layer_name]
            layers_dic[layer_name] = layer
            x = layer(tensor)
            
            if return_layers:
                return x, layers_dic
            
            return x

    return fn


def u_net(
    input_shape,
    nb_filters_0,
    output_channels,
    depth,
    sigma_noise,
    convlayer,
    updown_conv_sampling,
    unet_block_kwargs,
    unet_down_kwargs,
    unet_up_kwargs,
    channel_multiplication_factor=2,
    name=None,
):
    """Modular U-Net.

    Note that the dimensions of the input images should be
    multiples of 16.

    todo make this multichannel enabled
    """
    unet_block_kwargs = {
        **unet_block_kwargs,
        **dict(convlayer=convlayer)
    }

    unet_block = functools.partial(
        generic_unet_block, **unet_block_kwargs
    )

    unet_down_kwargs = {
        **unet_down_kwargs,
        **dict(
            conv_sampling=updown_conv_sampling,
            convlayer=convlayer,
        )
    }

    unet_down = functools.partial(
        generic_unet_down, **unet_down_kwargs
    )

    unet_up_kwargs = {
        **unet_up_kwargs,
        **dict(
            conv_sampling=updown_conv_sampling,
            convlayer=convlayer,
        )
    }

    unet_up = functools.partial(
        generic_unet_up, **unet_up_kwargs
    )

    x = x0 = layers.Input(input_shape, name="input")

    skips = {}
    for i in range(depth):
        nb_filters_begin = nb_filters_0 * channel_multiplication_factor ** i
        nb_filters_end = nb_filters_0 * channel_multiplication_factor ** (i + 1)
        x = unet_block(f"enc-block-{i}", nb_filters_1=nb_filters_begin, nb_filters_2=nb_filters_end)(x)
        skips[i] = x
        x = unet_down(nb_filters=nb_filters_end, name=f"enc-block-{i}")(x)

    nb_filters_begin = nb_filters_0 * channel_multiplication_factor ** depth
    nb_filters_end = nb_filters_0 * channel_multiplication_factor ** (depth + 1)
    x = unet_block(f"enc-block-{depth}", nb_filters_1=nb_filters_begin, nb_filters_2=nb_filters_end)(x)

    for i in reversed(range(depth)):
        nb_filters_up = nb_filters_0 * channel_multiplication_factor ** (i + 2)
        nb_filters_conv = nb_filters_0 * channel_multiplication_factor ** (i + 1)
        x = unet_up(nb_filters=nb_filters_up, name=f"dec-block-{i}")(x)
        x_skip = skips[i]
        x = layers.concatenate([x_skip, x], axis=-1)
        x = unet_block(f"dec-block-{i}", nb_filters_1=nb_filters_conv, nb_filters_2=nb_filters_conv)(x)

    if sigma_noise > 0:
        x = layers.GaussianNoise(sigma_noise, name="gaussian-noise")(x)

    if convlayer in (ConvLayer.conv2d, ConvLayer.conv2d_separable):
        x = layers.Conv2D(output_channels, 1, activation="softmax", name="out")(x)

    elif convlayer in (ConvLayer.conv3d, ConvLayer.conv3d_separable):
        x = layers.Conv3D(output_channels, 1, activation="softmax", name="out")(x)

    else:
        raise ValueError(f"{convlayer=}")

    return Model(x0, x, name=name)


kwargs_vanilla00 = dict(
    depth=4,
    sigma_noise=0,
    updown_conv_sampling=False,
    unet_block_kwargs=dict(
        kernel_size=3,
        res=False,
        batch_norm=False,
        dropout=0,
    ),
    unet_down_kwargs={},
    unet_up_kwargs={},
)

kwargs_vanilla01 = dict(
    depth=4,
    sigma_noise=0,
    updown_conv_sampling=False,
    unet_block_kwargs=dict(
        kernel_size=3,
        res=True,
        batch_norm=False,
        dropout=0,
    ),
    unet_down_kwargs={},
    unet_up_kwargs={},
)

kwargs_vanilla02 = dict(
    depth=4,
    sigma_noise=0,
    updown_conv_sampling=False,
    unet_block_kwargs=dict(
        kernel_size=3,
        res=True,
        batch_norm=True,
        dropout=0,
    ),
    unet_down_kwargs={},
    unet_up_kwargs={},
)

kwargs_vanilla03 = dict(
    depth=4,
    sigma_noise=0,
    updown_conv_sampling=True,
    unet_block_kwargs=dict(
        kernel_size=3,
        res=True,
        batch_norm=True,
        dropout=0,
    ),
    unet_down_kwargs=dict(
        batchnorm=True
    ),
    unet_up_kwargs=dict(
        batchnorm=True
    ),
)

kwargs_depth3 = dict(
    depth=3,
    sigma_noise=0,
    updown_conv_sampling=True,
    unet_block_kwargs=dict(
        kernel_size=3,
        res=True,
        batch_norm=True,
        dropout=0,
    ),
    unet_down_kwargs=dict(
        batchnorm=True
    ),
    unet_up_kwargs=dict(
        batchnorm=True
    ),
)


def u_net2halfd_IIencdec(
    input_shape,
    nb_filters_0,
    output_channels,
    depth,
    sigma_noise,
    convlayer,
    updown_conv_sampling,
    unet_block_kwargs,
    unet_down_kwargs,
    unet_up_kwargs,
    name=None,
):
    """
    todo make this multichannel enabled
    """
    assert convlayer in (ConvLayer.conv2d, ConvLayer.conv2d_separable), f"{convlayer=}"
    
    unet_block_kwargs = {
        **unet_block_kwargs,
        **dict(convlayer=convlayer),
        **dict(return_layers=True),
    }
    
    logger.debug(f"{dict2str(unet_block_kwargs)=}")
    
    unet_block = functools.partial(
        generic_unet_block, 
        **unet_block_kwargs,
    )

    unet_down_kwargs = {
        **unet_down_kwargs,
        **dict(
            conv_sampling=updown_conv_sampling,
            convlayer=convlayer,
        ),
        **dict(return_layers=True),
    }
    
    logger.debug(f"{dict2str(unet_down_kwargs)=}")

    unet_down = functools.partial(
        generic_unet_down, **unet_down_kwargs
    )

    unet_up_kwargs = {
        **unet_up_kwargs,
        **dict(
            conv_sampling=updown_conv_sampling,
            convlayer=convlayer,
        ),
        **dict(return_layers=True),
    }
    
    logger.debug(f"{dict2str(unet_up_kwargs)=}")

    unet_up = functools.partial(
        generic_unet_up, **unet_up_kwargs
    )
    
    nlayers = int(input_shape[-1])
    
    logger.debug(f"{nlayers=}")
    
    predicted_layer = nlayers // 2
    
    logger.debug(f"{predicted_layer=}")
    
    from tensorflow import slice as tf_slice

#     x = x0 = layers.Input(input_shape, name="input")
    x0 = layers.Input(input_shape, name="input")
    
    x0_splitted = [
        layers.Lambda(
            lambda x_: tf_slice(x_, (0, 0, 0, ch), (-1, -1, -1, 1))
        )(x0)
        for ch in range(nlayers)
    ]
    
    xs = x0_splitted
    
    skips = {}
    for i in range(depth):
        
        nb_filters_begin = nb_filters_0 * 2 ** i
        nb_filters_end = nb_filters_0 * 2 ** (i + 1)
        
        block_name = f"enc-block-{i}"
        block = unet_block(name=block_name, nb_filters_1=nb_filters_begin, nb_filters_2=nb_filters_end)
        
        block_name = f"enc-block-{i}"
        down_block = unet_down(nb_filters=nb_filters_end, name=block_name)
        
        block_reused_layers = None
        down_reused_layers = None
        
        for layer_idx in range(nlayers): 
            
            y, block_reused_layers = block(xs[layer_idx], reused_layers=block_reused_layers)
            xs[layer_idx] = y
            
            skips[(i, layer_idx)] = y
            
            y, down_reused_layers = down_block(xs[layer_idx], reused_layers=down_reused_layers)
            xs[layer_idx] = y

    nb_filters_begin = nb_filters_0 * 2 ** depth
    nb_filters_end = nb_filters_0 * 2 ** (depth + 1)
    
    block_name = f"enc-block-{depth}"
    block = unet_block(
        name=block_name, 
        nb_filters_1=nb_filters_begin, 
        nb_filters_2=nb_filters_end
    )
    
    block_reused_layers = None
    for layer_idx in range(nlayers):
        
        y, block_reused_layers = block(xs[layer_idx], reused_layers=block_reused_layers)
        xs[layer_idx] = y
            
#     layer_name = f"join-{depth}"
#     x = layers.concatenate(xs, axis=-1, name=layer_name)
    
#     nb_filters_begin = nlayers * (nb_filters_0 * 2 ** (depth + 1))
#     nb_filters_end = nlayers * (nb_filters_0 * 2 ** depth)
    
#     block_name = f"joined-enc-block-{depth}"
#     block = unet_block(
#         name=block_name, 
#         nb_filters_1=nb_filters_begin, 
#         nb_filters_2=nb_filters_end,
#     )
    
#     x, _ = block(x, reused_layers=None)

        
    for i in reversed(range(depth)):
        
        nb_filters_up = nb_filters_0 * 2 ** (i + 2)
        nb_filters_conv = nb_filters_0 * 2 ** (i + 1)
        
        block_name = f"dec-block-{i}"
        up_block = unet_up(
            nb_filters=nb_filters_up, 
            name=block_name,
        )
        block = unet_block(
            name=block_name, 
            nb_filters_1=nb_filters_conv, 
            nb_filters_2=nb_filters_conv,
        )
        
        up_reused_layers = None
        block_reused_layers = None
        
        for layer_idx in range(nlayers): 
            
            y, up_reused_layers = up_block(
                xs[layer_idx], 
                reused_layers=up_reused_layers
            )
            xs[layer_idx] = y
        
            x_skip = skips[(i, layer_idx)]
        
            xs[layer_idx] = layers.concatenate([x_skip, xs[layer_idx]], axis=-1, name=f"concat_{i}-layer_{layer_idx}")
        
            y, block_reused_layers = block(
                xs[layer_idx], 
                reused_layers=block_reused_layers,
            )
            xs[layer_idx] = y
            
    x = layers.concatenate(xs, name="join")

    if sigma_noise > 0:
        layer_name = "gaussian-noise"
        x = layers.GaussianNoise(sigma_noise, name=layer_name)(x)

    if convlayer in (ConvLayer.conv2d, ConvLayer.conv2d_separable):
        x = layers.Conv2D(output_channels, 1, activation="softmax", name="out")(x)

    elif convlayer in (ConvLayer.conv3d, ConvLayer.conv3d_separable):
        x = layers.Conv3D(output_channels, 1, activation="softmax", name="out")(x)

    else:
        raise ValueError(f"{convlayer=}")

    return Model(x0, x, name=name)


# just like `kwargs_vanilla03`
kwargs_IIencdec03 = dict(
    depth=4,
    sigma_noise=0,
    updown_conv_sampling=True,
    unet_block_kwargs=dict(
        kernel_size=3,
        res=True,
        batch_norm=True,
        dropout=0,
    ),
    unet_down_kwargs=dict(
        batchnorm=True
    ),
    unet_up_kwargs=dict(
        batchnorm=True
    ),
)

kwargs_IIencdec03_debug = {
    **kwargs_IIencdec03,
    **dict(
        depth=2, 
        updown_conv_sampling=False,
        unet_block_kwargs=dict(
            kernel_size=3,
            dropout=0,
            batch_norm=False, 
            res=False,
        )
    ),
}