from uvcgan.base.networks    import select_base_discriminator
from uvcgan.base.weight_init import init_weights
from uvcgan.torch.funcs      import prepare_model

def select_discriminator(name, **kwargs):
    return select_base_discriminator(name, **kwargs)

def construct_discriminator(model_config, image_shape, device):
    model = select_discriminator(
        model_config.model, image_shape = image_shape,
        **model_config.model_args
    )

    model = prepare_model(model, device)
    init_weights(model, model_config.weight_init)

    return model

