import os
import math
from itertools import islice

from uvcgan.config            import Args
from uvcgan.data              import get_data
from uvcgan.torch.funcs       import get_torch_device_smart, seed_everything
from uvcgan.cgan              import construct_model
from uvcgan.utils.model_state import ModelState

def slice_data_loader(loader, batch_size, n_samples = None):
    if n_samples is None:
        return (loader, len(loader))

    steps = min(math.ceil(n_samples / batch_size), len(loader))
    sliced_loader = islice(loader, steps)

    return (sliced_loader, steps)

def tensor_to_image(tensor):
    result = tensor.cpu().detach().numpy()

    if tensor.ndim == 4:
        result = result.squeeze(0)

    result = result.transpose((1, 2, 0))
    return result

def override_config(config, config_overrides):
    if config_overrides is None:
        return

    for (k,v) in config_overrides.items():
        config[k] = v

def get_evaldir(root, epoch, mkdir = False):
    if epoch is None:
        result = os.path.join(root, 'evals', 'final')
    else:
        result = os.path.join(root, 'evals', 'epoch_%04d' % epoch)

    if mkdir:
        os.makedirs(result, exist_ok = True)

    return result

def start_model_eval(path, epoch, model_state, **config_overrides):
    args   = Args.load(path)
    device = get_torch_device_smart()

    override_config(args.config, config_overrides)

    model = construct_model(
        args.savedir, args.config, is_train = False, device = device
    )

    if epoch == -1:
        epoch = max(model.find_last_checkpoint_epoch(), 0)

    print("Load checkpoint at epoch %s" % epoch)

    seed_everything(args.config.seed)
    model.load(epoch)

    model_state.set_model_state(model)
    evaldir = get_evaldir(path, epoch, mkdir = True)

    return (args, model, evaldir)

def load_eval_model_dset_from_cmdargs(cmdargs, **config_overrides):
    model_state = ModelState.from_str(cmdargs.model_state)

    args, model, evaldir = start_model_eval(
        cmdargs.model, cmdargs.epoch, model_state,
        batch_size = cmdargs.batch_size, **config_overrides
    )

    _, it_val = get_data(
        args.config.data, args.config.batch_size, args.workers
    )

    return (args, model, it_val, evaldir)

