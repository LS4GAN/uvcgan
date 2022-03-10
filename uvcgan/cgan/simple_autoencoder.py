from uvcgan.torch.select             import select_optimizer, select_loss
from uvcgan.torch.image_masking      import select_masking
from uvcgan.models.generator         import construct_generator

from .model_base import ModelBase

class SimpleAutoencoder(ModelBase):

    def _setup_images(self, _config):
        images = [ 'real', 'reco' ]

        if self.masking is not None:
            images.append('masked')

        for img_name in images:
            self.images[img_name] = None

    def _setup_models(self, config):
        self.models.encoder = construct_generator(
            config.generator, config.image_shape, self.device
        )

    def _setup_losses(self, config):
        self.losses['loss'] = None
        self.loss_fn = select_loss(config.loss)

        assert config.gradient_penalty is None, \
            "Autoencoder model does not support gradient penalty"

    def _setup_optimizers(self, config):
        self.optimizers.encoder = select_optimizer(
            self.models.encoder.parameters(), config.generator.optimizer
        )

    def __init__(
        self, savedir, config, is_train, device, masking = None
    ):
        # pylint: disable=too-many-arguments
        self.masking = select_masking(masking)
        super().__init__(savedir, config, is_train, device)

        assert config.discriminator is None, \
            "Autoencoder model does not use discriminator"

    def set_input(self, inputs):
        self.images.real = inputs[0].to(self.device)

    def forward(self):
        if self.masking is None:
            input_img = self.images.real
        else:
            self.images.masked = self.masking(self.images.real)
            input_img          = self.images.masked

        self.images.reco = self.models.encoder(input_img)

    def backward(self):
        loss = self.loss_fn(self.images.reco, self.images.real)
        loss.backward()

        self.losses.loss = loss

    def optimization_step(self):
        self.forward()

        for optimizer in self.optimizers.values():
            optimizer.zero_grad()

        self.backward()

        for optimizer in self.optimizers.values():
            optimizer.step()

