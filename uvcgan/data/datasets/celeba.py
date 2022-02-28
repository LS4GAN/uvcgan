import os
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from .funcs import load_images, sample_image

FNAME_ATTRS = 'list_attr_celeba.txt'
FNAME_SPLIT = 'list_eval_partition.txt'
SUBDIR_IMG  = 'img_align_celeba'

SPLIT_TRAIN = 'train'
SPLIT_VAL   = 'val'
SPLIT_TEST  = 'test'

SPLITS = {
    SPLIT_TRAIN : 0,
    SPLIT_VAL   : 1,
    SPLIT_TEST  : 2,
}

class CelebaDataset(Dataset):
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self, path,
        attr      = 'Young',
        split     = SPLIT_TRAIN,
        transform = None,
        seed      = None,
        **kwargs
    ):
        # pylint: disable=too-many-arguments
        assert split in SPLITS
        super().__init__(**kwargs)

        self._path      = path
        self._root_imgs = os.path.join(path, SUBDIR_IMG)
        self._split     = split
        self._attr      = attr
        self._prg       = None
        self._imgs_a    = []
        self._imgs_b    = []
        self._len       = 0
        self._transform = transform

        self.reseed(seed)
        self._collect_files()

    def reseed(self, seed):
        self._prg = np.random.default_rng(seed)

    def _collect_files(self):
        imgs_specs = CelebaDataset.load_image_specs(self._path)

        imgs_a, imgs_b = CelebaDataset.partition_images(
            imgs_specs, self._split, self._attr
        )

        self._imgs_a = [ os.path.join(self._root_imgs, x) for x in imgs_a ]
        self._imgs_b = [ os.path.join(self._root_imgs, x) for x in imgs_b ]

        self._len = max(len(self._imgs_a), len(self._imgs_b))

    @staticmethod
    def load_image_partition(root):
        path = os.path.join(root, FNAME_SPLIT)

        return pd.read_csv(
            path, sep = r'\s+', header = None, names = [ 'partition', ],
            index_col = 0
        )

    @staticmethod
    def load_image_attrs(root):
        path = os.path.join(root, FNAME_ATTRS)

        return pd.read_csv(
            path, sep = r'\s+', skiprows = 1, header = 0, index_col = 0
        )

    @staticmethod
    def load_image_specs(root):
        df_partition = CelebaDataset.load_image_partition(root)
        df_attrs     = CelebaDataset.load_image_attrs(root)

        return df_partition.join(df_attrs)

    @staticmethod
    def partition_images(image_specs, split, attr):
        part_mask = (image_specs.partition == SPLITS[split])

        if attr is None:
            imgs_a = image_specs[part_mask].index.to_list()
            imgs_b = []
        else:
            mask_a = (image_specs[attr] > 0)
            mask_b = ~mask_a

            imgs_a = image_specs[part_mask & mask_a].index.to_list()
            imgs_b = image_specs[part_mask & mask_b].index.to_list()

        return (imgs_a, imgs_b)

    def __len__(self):
        return self._len

    def _sample_image(self, images, index):
        return sample_image(
            images, index, self._prg, randomize = (self._split == SPLIT_TRAIN)
        )

    def __getitem__(self, index):
        paths = [ self._sample_image(self._imgs_a, index) ]

        if self._attr is not None:
            paths.append(self._sample_image(self._imgs_b, index))

        return load_images(paths, self._transform)

