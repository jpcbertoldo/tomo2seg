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
    
    
class NormLayer(Enum):
    none = 0
    batchnorm = 1
    layernorm = 2

    
class NormType:
    channel = 0
    spatial = 1
    spatial_channel = 2


def get_conv_layer(convlayer: ConvLayer):
    
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
    
    return conv_layer


def get_norm_layer(normlayer: NormLayer):
    
    if normlayer == NormLayer.none:
        norm_layer = None

    elif normlayer == NormLayer.batchnorm:
        norm_layer = layers.BatchNormalization

    elif normlayer == NormLayer.layernorm:
        norm_layer = layers.LayerNormalization

    else:
        raise ValueError(f"{normlayer=}")

    return norm_layer


def get_norm_axis(convlayer: ConvLayer, normtype: NormType):
    
    if normtype == NormType.channel:
        norm_axis = -1

    elif normtype == NormType.spatial:

        if convlayer in (ConvLayer.conv2d, ConvLayer.conv2d_separable):
            norm_axis = (1, 2)

        elif convlayer in (ConvLayer.conv3d, ConvLayer.conv3d_separable):
            norm_axis = (1, 2, 3)

        else:
            raise ValueError(f"{convlayer=}")

    elif normtype == NormType.spatial_channel:

        if convlayer in (ConvLayer.conv2d, ConvLayer.conv2d_separable):
            norm_axis = (1, 2, 3)

        elif convlayer in (ConvLayer.conv3d, ConvLayer.conv3d_separable):
            norm_axis = (1, 2, 3, 4)

        else:
            raise ValueError(f"{convlayer=}")

    else:
        raise ValueError(f"{norm_type=}")

    return norm_axis


# todo use model class instead
def generic_unet_block(
    name, 
    nb_filters_1, 
    nb_filters_2, 
    convlayer: ConvLayer, 
    normlayer: NormLayer,
    kernel_size, 
    res, 
    dropout, 
    normtype: NormType = NormType.channel,
    return_layers: bool = False,
    norm_kwargs=dict(),
):
    norm_layer = get_norm_layer(normlayer)
    
    if norm_layer is not None:
        
        norm_axis = get_norm_axis(convlayer, normtype)
        
    conv_layer = get_conv_layer(convlayer)
        
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

        if norm_layer is not None:
            layer_name = f"{name}-conv1-bn" if name is not None else None 
            layer = norm_layer(
                axis=norm_axis,
                name=layer_name,
                **norm_kwargs,
            )
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

        if norm_layer is not None:
            layer_name = f"{name}-conv2-bn" if name is not None else None 
            layer = norm_layer(
                axis=norm_axis,
                name=layer_name,
                **norm_kwargs,
            )
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

            if norm_layer is not None:
                layer_name = f"{name}-conv-skip-bn" if name is not None else None 
                layer = norm_layer(
                    axis=norm_axis,
                    name=layer_name,
                    **norm_kwargs,
                )
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
    conv_sampling, 
    convlayer, 
    nb_filters, 
    name, 
    normlayer: NormLayer,
    normtype: NormType = NormType.channel,
    return_layers: bool = False,
    norm_kwargs=dict(),
):
    
    if conv_sampling:
        
        norm_layer = get_norm_layer(normlayer)

        if norm_layer is not None:

            norm_axis = get_norm_axis(convlayer, normtype)

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
        
            if norm_layer is not None:
                layer_name = f"{name}-DOWN-bn" if name is not None else None
                layer = norm_layer(
                    axis=norm_axis,
                    name=layer_name,
                    **norm_kwargs,
                )
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
    conv_sampling, 
    convlayer, 
    nb_filters, 
    name,
    normlayer: NormLayer,
    normtype: NormType = NormType.channel,
    return_layers: bool = False,
    norm_kwargs=dict(),
):
    if conv_sampling:
        
        norm_layer = get_norm_layer(normlayer)

        if norm_layer is not None:

            norm_axis = get_norm_axis(convlayer, normtype)

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
        
            if norm_layer is not None:
                layer_name = f"{name}-UP-bn" if name is not None else None
                layer = norm_layer(
                    axis=norm_axis,
                    name=layer_name,
                    **norm_kwargs,
                )
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
        normlayer,
        normtype=NormType.channel,
        name=None,
        norm_kwargs=dict(),
):
    """Modular U-Net.

    Note that the dimensions of the input images should be
    multiples of 16.

    todo make this multichannel enabled
    """
    unet_block_kwargs = {
        **dict(
            convlayer=convlayer,
            normlayer=normlayer,
            normtype=normtype,
            norm_kwargs=norm_kwargs,
        ),
        **unet_block_kwargs,
    }

    logger.info(f"unet_block_kwargs\n{dict2str(unet_block_kwargs)}")
  
    unet_block = functools.partial(
        generic_unet_block, 
        **unet_block_kwargs
    )

    unet_down_kwargs = {
        **dict(
            conv_sampling=updown_conv_sampling,
            convlayer=convlayer,
            normlayer=normlayer,
            normtype=normtype,
            norm_kwargs=norm_kwargs,
        ),
        **unet_down_kwargs,
    }

    logger.info(f"unet_down_kwargs\n{dict2str(unet_down_kwargs)}")

    unet_down = functools.partial(
        generic_unet_down, 
        **unet_down_kwargs
    )

    unet_up_kwargs = {
        **dict(
            conv_sampling=updown_conv_sampling,
            convlayer=convlayer,
            normlayer=normlayer,
            normtype=normtype,
            norm_kwargs=norm_kwargs,
        ),
        **unet_up_kwargs,
    }

    logger.info(f"unet_up_kwargs\n{dict2str(unet_up_kwargs)}")
    
    unet_up = functools.partial(
        generic_unet_up, 
        **unet_up_kwargs,
    )

    x = x0 = layers.Input(input_shape, name="input")

    skips = {}
    for i in range(depth):
        nb_filters_begin = nb_filters_0 * 2 ** i
        nb_filters_end = nb_filters_0 * 2 ** (i + 1)
        x = unet_block(f"enc-block-{i}", nb_filters_1=nb_filters_begin, nb_filters_2=nb_filters_end)(x)
        skips[i] = x
        x = unet_down(nb_filters=nb_filters_end, name=f"enc-block-{i}")(x)

    nb_filters_begin = nb_filters_0 * 2 ** depth
    nb_filters_end = nb_filters_0 * 2 ** (depth + 1)
    x = unet_block(f"enc-block-{depth}", nb_filters_1=nb_filters_begin, nb_filters_2=nb_filters_end)(x)

    for i in reversed(range(depth)):
        nb_filters_up = nb_filters_0 * 2 ** (i + 2)
        nb_filters_conv = nb_filters_0 * 2 ** (i + 1)
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

