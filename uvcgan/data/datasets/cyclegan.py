import os
import numpy as np

from torch.utils.data import Dataset
from torchvision.datasets.folder import IMG_EXTENSIONS

from .funcs import load_images, sample_image

class CycleGANDataset(Dataset):
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self, path,
        align_train   = False,
        is_train      = False,
        seed          = None,
        transform     = None,
        **kwargs
    ):
        # pylint: disable=too-many-arguments
        super().__init__(**kwargs)

        if is_train:
            subdir_a = 'trainA'
            subdir_b = 'trainB'
        else:
            subdir_a = 'testA'
            subdir_b = 'testB'

        self._align_train = align_train
        self._is_train    = is_train
        self._path_a      = os.path.join(path, subdir_a)
        self._path_b      = os.path.join(path, subdir_b)
        self._imgs_a      = []
        self._imgs_b      = []
        self._transform   = transform
        self._len         = 0

        self.reseed(seed)
        self._collect_files()

    def reseed(self, seed):
        self._prg = np.random.default_rng(seed)

    @staticmethod
    def find_images_in_dir(path):
        extensions = set(IMG_EXTENSIONS)

        result = []
        for fname in os.listdir(path):
            fullpath = os.path.join(path, fname)

            if not os.path.isfile(fullpath):
                continue

            ext = os.path.splitext(fname)[1]
            if ext not in extensions:
                continue

            result.append(fullpath)

        result.sort()
        return result

    def _collect_files(self):
        self._imgs_a = CycleGANDataset.find_images_in_dir(self._path_a)
        self._imgs_b = CycleGANDataset.find_images_in_dir(self._path_b)

        self._len = max(len(self._imgs_a), len(self._imgs_b))

    def __len__(self):
        return self._len

    def _sample_image(self, images, index):
        randomize = (self._is_train and (not self._align_train))
        return sample_image(images, index, self._prg, randomize)

    def __getitem__(self, index):
        path_a = self._sample_image(self._imgs_a, index)
        path_b = self._sample_image(self._imgs_b, index)

        return load_images([path_a, path_b], self._transform)

