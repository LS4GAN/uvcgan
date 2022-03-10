
class ModelConfig:

    __slots__ = [
        'model',
        'model_args',
        'optimizer',
        'weight_init',
    ]

    def __init__(
        self,
        model,
        optimizer        = None,
        model_args       = None,
        weight_init      = None,
    ):
        self.model      = model
        self.model_args = model_args or {}
        self.optimizer  = optimizer or {
            'name' : 'AdamW', 'betas' : (0.5, 0.999), 'weight_decay' : 1e-5,
        }
        self.weight_init = weight_init

    def to_dict(self):
        return { x : getattr(self, x) for x in self.__slots__ }

