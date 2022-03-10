# pylint: disable=not-callable
# NOTE: Mistaken lint:
# E1102: self.encoder is not callable (not-callable)
from uvcgan.torch.select             import select_optimizer, select_loss
from uvcgan.torch.image_masking      import select_masking
from uvcgan.models.generator         import construct_generator

from .model_base import ModelBase

class Autoencoder(ModelBase):

    def _setup_images(self, _config):
        images = [ 'real_a', 'reco_a', 'real_b', 'reco_b', ]

        if self.masking is not None:
            images += [ 'masked_a', 'masked_b' ]

        for img_name in images:
            self.images[img_name] = None

    def _setup_models(self, config):
        if self.joint:
            self.models.encoder = construct_generator(
                config.generator, config.image_shape, self.device
            )
        else:
            self.models.encoder_a = construct_generator(
                config.generator, config.image_shape, self.device
            )
            self.models.encoder_b = construct_generator(
                config.generator, config.image_shape, self.device
            )

    def _setup_losses(self, config):
        losses = [ 'loss_a', 'loss_b' ]

        for loss in losses:
            self.losses[loss] = None

        self.loss_fn = select_loss(config.loss)

        assert config.gradient_penalty is None, \
            "Autoencoder model does not support gradient penalty"

    def _setup_optimizers(self, config):
        if self.joint:
            self.optimizers.encoder = select_optimizer(
                self.models.encoder.parameters(), config.generator.optimizer
            )
        else:
            self.optimizers.encoder_a = select_optimizer(
                self.models.encoder_a.parameters(), config.generator.optimizer
            )
            self.optimizers.encoder_b = select_optimizer(
                self.models.encoder_b.parameters(), config.generator.optimizer
            )

    def __init__(
        self, savedir, config, is_train, device,
        joint = False, masking = None
    ):
        # pylint: disable=too-many-arguments
        self.joint   = joint
        self.masking = select_masking(masking)

        super().__init__(savedir, config, is_train, device)

        assert config.discriminator is None, \
            "Autoencoder model does not use discriminator"

    def set_input(self, inputs):
        self.images.real_a = inputs[0].to(self.device)
        self.images.real_b = inputs[1].to(self.device)

    def forward(self):
        if self.masking is None:
            input_a = self.images.real_a
            input_b = self.images.real_b
        else:
            self.images.masked_a = self.masking(self.images.real_a)
            self.images.masked_b = self.masking(self.images.real_b)

            input_a = self.images.masked_a
            input_b = self.images.masked_b

        if self.joint:
            self.images.reco_a = self.models.encoder(input_a)
            self.images.reco_b = self.models.encoder(input_b)
        else:
            self.images.reco_a = self.models.encoder_a(input_a)
            self.images.reco_b = self.models.encoder_b(input_b)

    def backward_generator_base(self, real, reco):
        loss = self.loss_fn(reco, real)
        loss.backward()

        return loss

    def backward_generators(self):
        self.losses.loss_b = self.backward_generator_base(
            self.images.real_b, self.images.reco_b
        )

        self.losses.loss_a = self.backward_generator_base(
            self.images.real_a, self.images.reco_a
        )

    def optimization_step(self):
        self.forward()

        for optimizer in self.optimizers.values():
            optimizer.zero_grad()

        self.backward_generators()

        for optimizer in self.optimizers.values():
            optimizer.step()

