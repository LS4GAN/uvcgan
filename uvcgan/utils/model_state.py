from enum import Enum

LABEL_TRAIN = 'train'
LABEL_EVAL  = 'eval'

class ModelState(Enum):
    TRAIN = 0
    EVAL  = 1

    @staticmethod
    def from_str(state):
        if state == LABEL_TRAIN:
            return ModelState.TRAIN

        if state == LABEL_EVAL:
            return ModelState.EVAL

        raise ValueError(f"Unknown model state: '{state}'")

    def set_model_state(self, model):
        if self is ModelState.TRAIN:
            model.train()
        elif self is ModelState.EVAL:
            model.eval()
        else:
            raise NotImplementedError

    def __str__(self):
        if self is ModelState.TRAIN:
            return LABEL_TRAIN

        if self is ModelState.EVAL:
            return LABEL_EVAL

        raise NotImplementedError

