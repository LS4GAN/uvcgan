from .cyclegan           import CycleGANModel
from .pix2pix            import Pix2PixModel
from .autoencoder        import Autoencoder
from .simple_autoencoder import SimpleAutoencoder

def select_model(name, **kwargs):
    if name == 'cyclegan':
        return CycleGANModel(**kwargs)

    if name == 'pix2pix':
        return Pix2PixModel(**kwargs)

    if name == 'autoencoder':
        return Autoencoder(**kwargs)

    if name == 'simple-autoencoder':
        return SimpleAutoencoder(**kwargs)

    raise ValueError("Unknown model: %s" % name)

def construct_model(savedir, config, is_train, device):
    model = select_model(
        config.model, savedir = savedir, config = config, is_train = is_train,
        device = device, **config.model_args
    )

    return model

