# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes

from torch import nn

from uvcgan.torch.layers.transformer import PixelwiseViT
from uvcgan.torch.layers.unet        import UNet
from uvcgan.torch.select             import get_activ_layer

class ViTUNetGenerator(nn.Module):

    def __init__(
        self, features, n_heads, n_blocks, ffn_features, embed_features,
        activ, norm, image_shape, unet_features_list, unet_activ, unet_norm,
        unet_downsample = 'conv',
        unet_upsample   = 'upsample-conv',
        unet_rezero     = False,
        rezero          = True,
        activ_output    = None,
        **kwargs
    ):
        # pylint: disable = too-many-locals
        super().__init__(**kwargs)

        self.image_shape = image_shape

        self.net = UNet(
            unet_features_list, unet_activ, unet_norm, image_shape,
            unet_downsample, unet_upsample, unet_rezero
        )

        bottleneck = PixelwiseViT(
            features, n_heads, n_blocks, ffn_features, embed_features,
            activ, norm,
            image_shape = self.net.get_inner_shape(),
            rezero      = rezero
        )

        self.net.set_bottleneck(bottleneck)

        self.output = get_activ_layer(activ_output)

    def forward(self, x):
        # x : (N, C, H, W)
        result = self.net(x)

        return self.output(result)

