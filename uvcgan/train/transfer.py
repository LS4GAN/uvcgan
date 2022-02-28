import os
import logging

from uvcgan.consts import ROOT_OUTDIR
from uvcgan.config import Args
from uvcgan.cgan   import construct_model

LOGGER = logging.getLogger('uvcgan.train')

def load_base_model(model, transfer_config):
    try:
        model.load(epoch = None)
        return

    except IOError as e:
        if not transfer_config.allow_partial:
            raise IOError(
                "Failed to find fully trained model in '%s' for transfer: %s"\
                % (model.savedir, e)
            ) from e

    LOGGER.warning(
        (
            "Failed to find fully trained model in '%s' for transfer."
            " Trying to load from a checkpoint..."
        ), model.savedir
    )

    epoch = model.find_last_checkpoint_epoch()

    if epoch > 0:
        LOGGER.warning("Load transfer model from a checkpoint '%d'", epoch)
    else:
        raise RuntimeError("Failed to find transfer model checkpoints.")

    model.load(epoch)

def get_base_model(transfer_config, device):
    base_path = os.path.join(ROOT_OUTDIR, transfer_config.base_model)
    base_args = Args.load(base_path)

    model = construct_model(
        base_args.savedir, base_args.config, is_train = True, device = device
    )

    load_base_model(model, transfer_config)

    return model

def transfer_parameters(model, base_model, transfer_config):
    for (dst,src) in transfer_config.transfer_map.items():
        model.models[dst].load_state_dict(
            base_model.models[src].state_dict(),
            strict = transfer_config.strict
        )

def transfer(model, transfer_config):
    if transfer_config is None:
        return

    LOGGER.info(
        "Initiating parameter transfer : '%s'", transfer_config.to_dict()
    )

    base_model = get_base_model(transfer_config, model.device)
    transfer_parameters(model, base_model, transfer_config)


