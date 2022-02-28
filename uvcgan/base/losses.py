# LICENSE
# This file was extracted from
#   https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
# Please see `uvcgan/base/LICENSE` for copyright attribution and LICENSE

import torch
from torch import nn

class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(
        self, gan_mode, target_real_label = 1.0, target_fake_label = 0.0
    ):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) -- the type of GAN objective.
                Choices: vanilla, lsgan, and wgangp.
            target_real_label (bool) -- label for a real image
            target_fake_label (bool) -- label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. Vanilla GANs will handle it with
        BCEWithLogitsLoss.
        """
        super().__init__()

        # pylint: disable=not-callable
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'wgan':
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) -- tpyically the prediction from a
                discriminator
            target_is_real (bool) -- if the ground truth label is for real
                images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of
            the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def forward(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) -- tpyically the prediction output from a
                discriminator
            target_is_real (bool) -- if the ground truth label is for real
                images or fake images

        Returns:
            the calculated loss.
        """

        if self.gan_mode == 'wgan':
            if target_is_real:
                return -prediction.mean()
            else:
                return prediction.mean()

        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)

# pylint: disable=too-many-arguments
# pylint: disable=redefined-builtin
def cal_gradient_penalty(
    netD, real_data, fake_data, device,
    type = 'mixed', constant = 1.0, lambda_gp = 10.0
):
    """Calculate the gradient penalty loss, used in WGAN-GP

    source: https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- torch device
        type (str)                  -- if we mix real and fake data or not
            Choices: [real | fake | mixed].
        constant (float)            -- the constant used in formula:
            (||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp == 0.0:
        return 0.0, None

    if type == 'real':
        interpolatesv = real_data
    elif type == 'fake':
        interpolatesv = fake_data
    elif type == 'mixed':
        alpha = torch.rand(real_data.shape[0], 1, device = device)
        alpha = alpha.expand(
            real_data.shape[0], real_data.nelement() // real_data.shape[0]
        ).contiguous().view(*real_data.shape)

        interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
    else:
        raise NotImplementedError('{} not implemented'.format(type))

    interpolatesv.requires_grad_(True)
    disc_interpolates = netD(interpolatesv)

    gradients = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolatesv,
        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
        create_graph=True, retain_graph=True, only_inputs=True
    )

    gradients = gradients[0].view(real_data.size(0), -1)

    gradient_penalty = (
        ((gradients + 1e-16).norm(2, dim=1) - constant) ** 2
    ).mean() * lambda_gp

    return gradient_penalty, gradients

