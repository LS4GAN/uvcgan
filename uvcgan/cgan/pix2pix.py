# LICENSE
# This file was extracted from
#   https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
# Please see `uvcgan/base/LICENSE` for copyright attribution and LICENSE

# pylint: disable=not-callable
# NOTE: Mistaken lint:
# E1102: self.criterion_gan is not callable (not-callable)

import torch

from uvcgan.torch.select         import select_optimizer
from uvcgan.base.losses          import GANLoss, cal_gradient_penalty
from uvcgan.models.discriminator import construct_discriminator
from uvcgan.models.generator     import construct_generator

from .model_base import ModelBase

class Pix2PixModel(ModelBase):

    def _setup_images(self, _config):
        images = [ 'real_a', 'fake_b', 'real_b', 'fake_a', ]

        for img_name in images:
            self.images[img_name] = None

    def _setup_models(self, config):
        self.models.gen_ab = construct_generator(
            config.generator, config.image_shape, self.device
        )
        self.models.gen_ba = construct_generator(
            config.generator, config.image_shape, self.device
        )

        if self.is_train:
            extended_image_shape = (
                2 * config.image_shape[0], *config.image_shape[1:]
            )

            self.models.disc_a = construct_discriminator(
                config.discriminator, extended_image_shape, self.device
            )
            self.models.disc_b = construct_discriminator(
                config.discriminator, extended_image_shape, self.device
            )

    def _setup_losses(self, config):
        losses = [ 'gen_ab', 'gen_ba', 'l1_ab', 'l1_ba', 'disc_a', 'disc_b' ]

        for loss in losses:
            self.losses[loss] = None

    def _setup_optimizers(self, config):
        self.optimizers.gen_ab = select_optimizer(
            self.models.gen_ab.parameters(), config.generator.optimizer
        )
        self.optimizers.gen_ba = select_optimizer(
            self.models.gen_ba.parameters(), config.generator.optimizer
        )

        self.optimizers.disc_a = select_optimizer(
            self.models.disc_a.parameters(), config.discriminator.optimizer
        )
        self.optimizers.disc_b = select_optimizer(
            self.models.disc_b.parameters(), config.discriminator.optimizer
        )

    def __init__(self, savedir, config, is_train, device):
        super().__init__(savedir, config, is_train, device)

        self.criterion_gan    = GANLoss(config.loss).to(self.device)
        self.criterion_l1     = torch.nn.L1Loss()
        self.gradient_penalty = config.gradient_penalty

    def set_input(self, inputs):
        self.images.real_a = inputs[0].to(self.device)
        self.images.real_b = inputs[1].to(self.device)

    def forward(self):
        self.images.fake_b = self.models.gen_ab(self.images.real_a)
        self.images.fake_a = self.models.gen_ba(self.images.real_b)

    def backward_discriminator_base(self, model, real, fake, preimage):
        cond_real = torch.cat([real, preimage], dim = 1)
        cond_fake = torch.cat([fake, preimage], dim = 1).detach()

        pred_real = model(cond_real)
        loss_real = self.criterion_gan(pred_real, True)

        pred_fake = model(cond_fake)
        loss_fake = self.criterion_gan(pred_fake, False)

        loss = (loss_real + loss_fake) * 0.5

        if self.gradient_penalty is not None:
            loss += cal_gradient_penalty(
                model, cond_real, cond_fake, real.device,
                **self.gradient_penalty
            )[0]

        loss.backward()
        return loss

    def backward_discriminators(self):
        self.losses.disc_b = self.backward_discriminator_base(
            self.models.disc_b,
            self.images.real_b, self.images.fake_b, self.images.real_a
        )

        self.losses.disc_a = self.backward_discriminator_base(
            self.models.disc_a,
            self.images.real_a, self.images.fake_a, self.images.real_b
        )

    def backward_generator_base(self, disc, real, fake, preimage):
        loss_gen = self.criterion_gan(
            disc(torch.cat([fake, preimage], dim = 1)), True
        )

        loss_l1 = self.criterion_l1(fake, real)

        loss = loss_gen + loss_l1
        loss.backward()

        return (loss_gen, loss_l1)

    def backward_generators(self):
        self.losses.gen_ab, self.losses.l1_ab = self.backward_generator_base(
            self.models.disc_b,
            self.images.real_b, self.images.fake_b, self.images.real_a
        )

        self.losses.gen_ba, self.losses.l1_ba = self.backward_generator_base(
            self.models.disc_a,
            self.images.real_a, self.images.fake_a, self.images.real_b
        )

    def optimization_step(self):
        self.forward()

        # Generators
        self.set_requires_grad([self.models.disc_a, self.models.disc_b], False)
        self.optimizers.gen_ab.zero_grad()
        self.optimizers.gen_ba.zero_grad()
        self.backward_generators()
        self.optimizers.gen_ab.step()
        self.optimizers.gen_ba.step()

        # Discriminators
        self.set_requires_grad([self.models.disc_a, self.models.disc_b], True)
        self.optimizers.disc_a.zero_grad()
        self.optimizers.disc_b.zero_grad()
        self.backward_discriminators()
        self.optimizers.disc_a.step()
        self.optimizers.disc_b.step()

