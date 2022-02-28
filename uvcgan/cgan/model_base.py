# LICENSE
# This file was extracted from
#   https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
# Please see `uvcgan/base/LICENSE` for copyright attribution and LICENSE

import logging
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from uvcgan.base.schedulers import get_scheduler
from .named_dict import NamedDict
from .checkpoint import find_last_checkpoint_epoch, save, load

PREFIX_MODEL = 'net'
PREFIX_OPT   = 'opt'
PREFIX_SCHED = 'sched'

LOGGER = logging.getLogger('uvcgan.cgan')

class ModelBase:
    # pylint: disable=too-many-instance-attributes

    def __init__(self, savedir, config, is_train, device):
        self.is_train     = is_train
        self.device       = device
        self.savedir      = savedir

        self.models       = NamedDict()
        self.images       = NamedDict()
        self.optimizers   = NamedDict()
        self.schedulers   = NamedDict()
        self.losses       = NamedDict()
        self.metric       = 0
        self.epoch        = 0

        self._setup_images(config)
        self._setup_models(config)
        self._setup_losses(config)

        if is_train:
            self._setup_optimizers(config)
            self._setup_schedulers(config)

    def set_input(self, inputs):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def optimization_step(self):
        raise NotImplementedError

    def _setup_images(self, config):
        raise NotImplementedError

    def _setup_models(self, config):
        raise NotImplementedError

    def _setup_losses(self, config):
        raise NotImplementedError

    def _setup_optimizers(self, config):
        raise NotImplementedError

    def _setup_schedulers(self, config):
        for (name,opt) in self.optimizers.items():
            self.schedulers[name] = get_scheduler(opt, config.scheduler)

    def _save_model_state(self, epoch):
        pass

    def _load_model_state(self, epoch):
        pass

    def _handle_epoch_end(self, epoch):
        pass

    def eval(self):
        for model in self.models.values():
            model.eval()

    def train(self):
        for model in self.models.values():
            model.train()

    def forward_nograd(self):
        with torch.no_grad():
            self.forward()

    def find_last_checkpoint_epoch(self):
        return find_last_checkpoint_epoch(self.savedir, PREFIX_MODEL)

    def load(self, epoch):
        if (epoch is not None) and (epoch <= 0):
            return

        LOGGER.debug('Loading model from epoch %s', epoch)

        load(self.models,     self.savedir, PREFIX_MODEL, epoch, self.device)
        load(self.optimizers, self.savedir, PREFIX_OPT,   epoch, self.device)
        load(self.schedulers, self.savedir, PREFIX_SCHED, epoch, self.device)

        self.epoch = epoch
        self._load_model_state(epoch)

    def save(self, epoch = None):
        LOGGER.debug('Saving model at epoch %s', epoch)

        save(self.models,     self.savedir, PREFIX_MODEL, epoch)
        save(self.optimizers, self.savedir, PREFIX_OPT,   epoch)
        save(self.schedulers, self.savedir, PREFIX_SCHED, epoch)

        self._save_model_state(epoch)

    def end_epoch(self, epoch = None):
        for scheduler in self.schedulers.values():
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(self.metric)
            else:
                scheduler.step()

        self._handle_epoch_end(epoch)

        if epoch is None:
            self.epoch = self.epoch + 1
        else:
            self.epoch = epoch

    def pprint(self, verbose):
        for name,model in self.models.items():
            num_params = 0

            for param in model.parameters():
                num_params += param.numel()

            if verbose:
                print(model)

            print(
                '[Network %s] Total number of parameters : %.3f M' % (
                    name, num_params / 1e6
                )
            )

    def set_requires_grad(self, models, requires_grad = False):
        # pylint: disable=no-self-use
        if not isinstance(models, list):
            models = [models, ]

        for model in models:
            for param in model.parameters():
                param.requires_grad = requires_grad

    def get_current_losses(self):
        result = {}

        for (k,v) in self.losses.items():
            result[k] = float(v)

        return result

