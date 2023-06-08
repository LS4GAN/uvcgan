from PIL import Image

from .funcs    import apply_if_not_none
from .cyclegan import CycleGANDataset

def image_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)

        n_channels = len(img.getbands())

        if n_channels == 1:
            return img.copy()

        return img.convert("RGB")

def load_images_v2(paths, transform = None):
    result = [ apply_if_not_none(image_loader, x) for x in paths ]

    if transform is not None:
        result = [ apply_if_not_none(transform, x) for x in result ]

    return result

class CycleGANv2Dataset(CycleGANDataset):
    # pylint: disable=too-many-instance-attributes

    def __getitem__(self, index):
        path_a = self._sample_image(self._imgs_a, index)
        path_b = self._sample_image(self._imgs_b, index)

        return load_images_v2([path_a, path_b], self._transform)

