# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes

import torch
from torch import nn

from uvcgan.torch.select import get_norm_layer, get_activ_layer

from .cnn import get_downsample_x2_layer, get_upsample_x2_layer

class UnetBasicBlock(nn.Module):

    def __init__(
        self, in_features, out_features, activ, norm, mid_features = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        if mid_features is None:
            mid_features = out_features

        self.block = nn.Sequential(
            get_norm_layer(norm, in_features),
            nn.Conv2d(in_features, mid_features, kernel_size = 3, padding = 1),
            get_activ_layer(activ),

            get_norm_layer(norm, mid_features),
            nn.Conv2d(
                mid_features, out_features, kernel_size = 3, padding = 1
            ),
            get_activ_layer(activ),
        )

    def forward(self, x):
        return self.block(x)

class UNetEncBlock(nn.Module):

    def __init__(
        self, features, activ, norm, downsample, input_shape, **kwargs
    ):
        super().__init__(**kwargs)

        self.downsample, output_features = \
            get_downsample_x2_layer(downsample, features)

        (C, H, W)  = input_shape
        self.block = UnetBasicBlock(C, features, activ, norm)

        self.output_shape = (output_features, H//2, W//2)

    def get_output_shape(self):
        return self.output_shape

    def forward(self, x):
        r = self.block(x)
        y = self.downsample(r)
        return (y, r)

class UNetDecBlock(nn.Module):

    def __init__(
        self, output_shape, activ, norm, upsample, input_shape,
        rezero = True, **kwargs
    ):
        super().__init__(**kwargs)

        self.upsample, input_features = get_upsample_x2_layer(
            upsample, input_shape[0]
        )

        self.block = UnetBasicBlock(
            2 * input_features, output_shape[0], activ, norm,
            mid_features = max(input_features, input_shape[0])
        )

        if rezero:
            self.re_alpha = nn.Parameter(torch.zeros((1, )))
        else:
            self.re_alpha = 1

    def forward(self, x, r):
        # x : (N, C, H_in, W_in)
        # r : (N, C, H_out, W_out)

        # x : (N, C_up, H_out, W_out)
        x = self.re_alpha * self.upsample(x)

        # y : (N, C + C_up, H_out, W_out)
        y = torch.cat([x, r], dim = 1)

        # result : (N, C_out, H_out, W_out)
        return self.block(y)

    def extra_repr(self):
        return 're_alpha = %e' % (self.re_alpha, )

class UNetBlock(nn.Module):

    def __init__(
        self, features, activ, norm, image_shape, downsample, upsample,
        rezero = True, **kwargs
    ):
        super().__init__(**kwargs)

        self.conv = UNetEncBlock(
            features, activ, norm, downsample, image_shape
        )

        self.inner_shape  = self.conv.get_output_shape()
        self.inner_module = None

        self.deconv = UNetDecBlock(
            image_shape, activ, norm, upsample, self.inner_shape, rezero
        )

    def get_inner_shape(self):
        return self.inner_shape

    def set_inner_module(self, module):
        self.inner_module = module

    def get_inner_module(self):
        return self.inner_module

    def forward(self, x):
        # x : (N, C, H, W)

        # y : (N, C_inner, H_inner, W_inner)
        # r : (N, C_inner, H, W)
        (y, r) = self.conv(x)

        # y : (N, C_inner, H_inner, W_inner)
        y = self.inner_module(y)

        # y : (N, C, H, W)
        y = self.deconv(y, r)

        return y

class UNet(nn.Module):

    def __init__(
        self, features_list, activ, norm, image_shape, downsample, upsample,
        rezero = True, **kwargs
    ):
        # pylint: disable = too-many-locals
        super().__init__(**kwargs)

        self.features_list = features_list
        self.image_shape   = image_shape

        self._construct_input_layer(activ)
        self._construct_output_layer()

        unet_layers = []
        curr_image_shape = (features_list[0], *image_shape[1:])

        for features in features_list:
            layer = UNetBlock(
                features, activ, norm, curr_image_shape, downsample, upsample,
                rezero
            )
            curr_image_shape = layer.get_inner_shape()
            unet_layers.append(layer)

        for idx in range(len(unet_layers)-1):
            unet_layers[idx].set_inner_module(unet_layers[idx+1])

        self.unet = unet_layers[0]

    def _construct_input_layer(self, activ):
        self.layer_input = nn.Sequential(
            nn.Conv2d(
                self.image_shape[0], self.features_list[0],
                kernel_size = 3, padding = 1
            ),
            get_activ_layer(activ),
        )

    def _construct_output_layer(self):
        self.layer_output = nn.Conv2d(
            self.features_list[0], self.image_shape[0], kernel_size = 1
        )

    def get_innermost_block(self):
        result = self.unet

        for _ in range(len(self.features_list)-1):
            result = result.get_inner_module()

        return result

    def set_bottleneck(self, module):
        self.get_innermost_block().set_inner_module(module)

    def get_bottleneck(self):
        return self.get_innermost_block().get_inner_module()

    def get_inner_shape(self):
        return self.get_innermost_block().get_inner_shape()

    def forward(self, x):
        # x : (N, C, H, W)

        y = self.layer_input(x)
        y = self.unet(y)
        y = self.layer_output(y)

        return y

