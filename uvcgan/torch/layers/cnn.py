from torch import nn
from uvcgan.torch.select import extract_name_kwargs

def get_downsample_x2_conv2_layer(features, **kwargs):
    return (
        nn.Conv2d(features, features, kernel_size = 2, stride = 2, **kwargs),
        features
    )

def get_downsample_x2_conv3_layer(features, **kwargs):
    return (
        nn.Conv2d(
            features, features, kernel_size = 3, stride = 2, padding = 1,
            **kwargs
        ),
        features
    )

def get_downsample_x2_pixelshuffle_layer(features, **kwargs):
    out_features = 4 * features
    return (nn.PixelUnshuffle(downscale_factor = 2, **kwargs), out_features)

def get_downsample_x2_pixelshuffle_conv_layer(features, **kwargs):
    out_features = features * 4

    layer = nn.Sequential(
        nn.PixelUnshuffle(downscale_factor = 2, **kwargs),
        nn.Conv2d(
            out_features, out_features, kernel_size = 3, padding = 1
        ),
    )

    return (layer, out_features)

def get_upsample_x2_deconv2_layer(features, **kwargs):
    return (
        nn.ConvTranspose2d(
            features, features, kernel_size = 2, stride = 2, **kwargs
        ),
        features
    )

def get_upsample_x2_upconv_layer(features, **kwargs):
    layer = nn.Sequential(
        nn.Upsample(scale_factor = 2, **kwargs),
        nn.Conv2d(features, features, kernel_size = 3, padding = 1),
    )

    return (layer, features)

def get_upsample_x2_pixelshuffle_conv_layer(features, **kwargs):
    out_features = features // 4

    layer = nn.Sequential(
        nn.PixelShuffle(upscale_factor = 2, **kwargs),
        nn.Conv2d(out_features, out_features, kernel_size = 3, padding = 1),
    )

    return (layer, out_features)

def get_downsample_x2_layer(layer, features):
    name, kwargs = extract_name_kwargs(layer)

    if name == 'conv':
        return get_downsample_x2_conv2_layer(features, **kwargs)

    if name == 'conv3':
        return get_downsample_x2_conv3_layer(features, **kwargs)

    if name == 'avgpool':
        return (nn.AvgPool2d(kernel_size = 2, stride = 2, **kwargs), features)

    if name == 'maxpool':
        return (nn.MaxPool2d(kernel_size = 2, stride = 2, **kwargs), features)

    if name == 'pixel-unshuffle':
        return get_downsample_x2_pixelshuffle_layer(features, **kwargs)

    if name == 'pixel-unshuffle-conv':
        return get_downsample_x2_pixelshuffle_conv_layer(features, **kwargs)

    raise ValueError("Unknown Downsample Layer: '%s'" % name)

def get_upsample_x2_layer(layer, features):
    name, kwargs = extract_name_kwargs(layer)

    if name == 'deconv':
        return get_upsample_x2_deconv2_layer(features, **kwargs)

    if name == 'upsample':
        return (nn.Upsample(scale_factor = 2, **kwargs), features)

    if name == 'upsample-conv':
        return get_upsample_x2_upconv_layer(features, **kwargs)

    if name == 'pixel-shuffle':
        return (nn.PixelShuffle(upscale_factor = 2, **kwargs), features // 4)

    if name == 'pixel-shuffle-conv':
        return get_upsample_x2_pixelshuffle_conv_layer(features, **kwargs)

    raise ValueError("Unknown Upsample Layer: '%s'" % name)

