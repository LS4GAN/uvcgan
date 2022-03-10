from .config_base     import ConfigBase

class DataConfig(ConfigBase):
    """Data configuration.

    Parameters
    ----------
    dataset : str
        Name of the dataset to use.
    dataset_args : dict or None
        Optional arguments that will be passed to the dataset constructor.
        Default: None.
    transform_train : None or str or dict or list of those
        Transformations to be applied to the training dataset.
        If `transform_train` is None, then no transformations will be applied
        to the training dataset.
        If `transform_train` is str, then its value is interpreted as a name
        of the transformation.
        If `transform_train` is dict, then it is expected to be of the form
        `{ 'name' : TRANFORM_NAME, **kwargs }`, where 'name' is the name of
        the transformation, and `kwargs` dict will be passed to the
        transformation constructor.
        Otherwise, `transform_train` is expected to be a list of values above.
        The corresponding transformations will be chained together in the
        order that they are specified.
        Default: None.
    transform_val : None or str or dict or list of those
        Transformations to be applied to the validation dataset.
        C.f. `transform_train`.
        Default: None.
    """

    __slots__ = [
        'dataset',
        'dataset_args',
        'transform_train',
        'transform_val',
    ]

    def __init__(
        self,
        dataset,
        dataset_args    = None,
        transform_train = None,
        transform_val   = None,
    ):
        # pylint: disable=too-many-arguments
        self.dataset         = dataset
        self.dataset_args    = dataset_args or {}
        self.transform_train = transform_train
        self.transform_val   = transform_val

