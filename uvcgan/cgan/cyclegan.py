# LICENSE
# This file was extracted from
#   https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
# Please see `uvcgan/base/LICENSE` for copyright attribution and LICENSE

# pylint: disable=not-callable
# NOTE: Mistaken lint:
# E1102: self.criterion_gan is not callable (not-callable)

import itertools
import torch

from uvcgan.torch.select         import select_optimizer
from uvcgan.base.image_pool      import ImagePool
from uvcgan.base.losses          import GANLoss, cal_gradient_penalty
from uvcgan.models.discriminator import construct_discriminator
from uvcgan.models.generator     import construct_generator

from .model_base import ModelBase

class CycleGANModel(ModelBase):
    # pylint: disable=too-many-instance-attributes

    def _setup_images(self, _config):
        images = [ 'real_a', 'fake_b', 'reco_a', 'real_b', 'fake_a', 'reco_b' ]

        if self.is_train and self.lambda_idt > 0:
            images += [ 'idt_a', 'idt_b' ]

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
            self.models.disc_a = construct_discriminator(
                config.discriminator, config.image_shape, self.device
            )
            self.models.disc_b = construct_discriminator(
                config.discriminator, config.image_shape, self.device
            )

    def _setup_losses(self, config):
        losses = [
            'gen_ab', 'gen_ba', 'cycle_a', 'cycle_b', 'disc_a', 'disc_b'
        ]

        if self.is_train and self.lambda_idt > 0:
            losses += [ 'idt_a', 'idt_b' ]

        for loss in losses:
            self.losses[loss] = None

    def _setup_optimizers(self, config):
        self.optimizers.gen = select_optimizer(
            itertools.chain(
                self.models.gen_ab.parameters(),
                self.models.gen_ba.parameters()
            ),
            config.generator.optimizer
        )

        self.optimizers.disc = select_optimizer(
            itertools.chain(
                self.models.disc_a.parameters(),
                self.models.disc_b.parameters()
            ),
            config.discriminator.optimizer
        )

    def __init__(
        self, savedir, config, is_train, device, pool_size = 50,
        lambda_a = 10.0, lambda_b = 10.0, lambda_idt = 0.5
    ):
        # pylint: disable=too-many-arguments
        self.lambda_a   = lambda_a
        self.lambda_b   = lambda_b
        self.lambda_idt = lambda_idt

        super().__init__(savedir, config, is_train, device)

        self.criterion_gan    = GANLoss(config.loss).to(self.device)
        self.gradient_penalty = config.gradient_penalty
        self.criterion_cycle  = torch.nn.L1Loss()
        self.criterion_idt    = torch.nn.L1Loss()

        if self.is_train:
            self.pred_a_pool = ImagePool(pool_size)
            self.pred_b_pool = ImagePool(pool_size)

    def set_input(self, inputs):
        def maybe_get(batch, device):
            if batch is None:
                return None

            return batch.to(device)

        self.images.real_a = maybe_get(inputs[0], self.device)
        self.images.real_b = maybe_get(inputs[1], self.device)

    def forward(self):
        def simple_fwd(batch, gen_fwd, gen_bkw):
            if batch is None:
                return (None, None)

            fake = gen_fwd(batch)
            reco = gen_bkw(fake)

            return (fake, reco)

        self.images.fake_b, self.images.reco_a = simple_fwd(
            self.images.real_a, self.models.gen_ab, self.models.gen_ba
        )

        self.images.fake_a, self.images.reco_b = simple_fwd(
            self.images.real_b, self.models.gen_ba, self.models.gen_ab
        )

    def backward_discriminator_base(self, model, real, fake):
        pred_real = model(real)
        loss_real = self.criterion_gan(pred_real, True)

        #
        # NOTE:
        #   This is a workaround to a pytorch 1.9.0 bug that manifests when
        #   cudnn is enabled. When the bug is solved remove no_grad block and
        #   replace `model(fake)` by `model(fake.detach())`.
        #
        #   bug: https://github.com/pytorch/pytorch/issues/48439
        #
        with torch.no_grad():
            fake = fake.contiguous()

        pred_fake = model(fake)
        loss_fake = self.criterion_gan(pred_fake, False)

        loss = (loss_real + loss_fake) * 0.5

        if self.gradient_penalty is not None:
            loss += cal_gradient_penalty(
                model, real, fake, real.device, **self.gradient_penalty
            )[0]

        loss.backward()
        return loss

    def backward_discriminators(self):
        fake_a = self.pred_a_pool.query(self.images.fake_a)
        fake_b = self.pred_b_pool.query(self.images.fake_b)

        self.losses.disc_b = self.backward_discriminator_base(
            self.models.disc_b, self.images.real_b, fake_b
        )

        self.losses.disc_a = self.backward_discriminator_base(
            self.models.disc_a, self.images.real_a, fake_a
        )

    def backward_generators(self):
        lambda_idt = self.lambda_idt
        lambda_a   = self.lambda_a
        lambda_b   = self.lambda_b

        self.losses.gen_ab = self.criterion_gan(
            self.models.disc_b(self.images.fake_b), True
        )
        self.losses.gen_ba = self.criterion_gan(
            self.models.disc_a(self.images.fake_a), True
        )
        self.losses.cycle_a = lambda_a * self.criterion_cycle(
            self.images.reco_a, self.images.real_a
        )
        self.losses.cycle_b = lambda_b * self.criterion_cycle(
            self.images.reco_b, self.images.real_b
        )

        loss = (
              self.losses.gen_ab  + self.losses.gen_ba
            + self.losses.cycle_a + self.losses.cycle_b
        )

        if lambda_idt > 0:
            self.images.idt_b = self.models.gen_ab(self.images.real_b)
            self.losses.idt_b = lambda_b * lambda_idt * self.criterion_idt(
                self.images.idt_b, self.images.real_b
            )

            self.images.idt_a = self.models.gen_ba(self.images.real_a)
            self.losses.idt_a = lambda_a * lambda_idt * self.criterion_idt(
                self.images.idt_a, self.images.real_a
            )

            loss += (self.losses.idt_a + self.losses.idt_b)

        loss.backward()

    def optimization_step(self):
        self.forward()

        # Generators
        self.set_requires_grad([self.models.disc_a, self.models.disc_b], False)
        self.optimizers.gen.zero_grad()
        self.backward_generators()
        self.optimizers.gen.step()

        # Discriminators
        self.set_requires_grad([self.models.disc_a, self.models.disc_b], True)
        self.optimizers.disc.zero_grad()
        self.backward_discriminators()
        self.optimizers.disc.step()

