import difflib
import os
from .config import Config

LABEL_FNAME = 'label'

def get_config_difference(config_old, config_new):
    diff_gen = difflib.unified_diff(
        config_old.to_json(sort_keys = True, indent = 4).split('\n'),
        config_new.to_json(sort_keys = True, indent = 4).split('\n'),
        fromfile = 'Old Config',
        tofile   = 'New Config',
    )

    return "\n".join(diff_gen)

class Args:
    __slots__ = [
        'config',
        'label',
        'savedir',
        'checkpoint',
        'log_level',
        'workers',
    ]

    def __init__(
        self, config, savedir, label,
        log_level  = 'INFO',
        checkpoint = 100,
        workers    = None
    ):
        # pylint: disable=too-many-arguments
        self.config     = config
        self.label      = label
        self.savedir    = savedir
        self.checkpoint = checkpoint
        self.log_level  = log_level
        self.workers    = workers

    def __getattr__(self, attr):
        return getattr(self.config, attr)

    def save(self):
        self.config.save(self.savedir)

        if self.label is not None:
            # pylint: disable=unspecified-encoding
            with open(os.path.join(self.savedir, LABEL_FNAME), 'wt') as f:
                f.write(self.label)

    def check_no_collision(self):
        try:
            old_config = Config.load(self.savedir)
        except IOError:
            return

        old = old_config.to_json(sort_keys = True)
        new = self.config.to_json(sort_keys = True)

        if old != new:
            diff = get_config_difference(old_config, self.config)

            raise RuntimeError(
                (
                    "Config collision detected in '%s'."
                    " Current config\n%s\n"
                    "does not match saved config\n%s\n"
                    "Difference:\n%s"
                ) % (self.savedir, new, old, diff)
            )

    @staticmethod
    def from_args_dict(
        outdir,
        label      = None,
        log_level  = 'INFO',
        checkpoint = 100,
        workers    = None,
        **args_dict
    ):
        config  = Config(**args_dict)
        savedir = config.get_savedir(outdir, label)

        result = Args(config, savedir, label, log_level, checkpoint, workers)
        result.check_no_collision()

        result.save()

        return result

    @staticmethod
    def load(savedir):
        config = Config.load(savedir)
        label  = None

        label_path = os.path.join(savedir, LABEL_FNAME)

        if os.path.exists(label_path):
            # pylint: disable=unspecified-encoding
            with open(label_path, 'rt') as f:
                label = f.read()

        return Args(config, savedir, label)

