from torchvision.datasets.folder import default_loader

def sample_image(images, index, prg, randomize = False):
    if randomize:
        return prg.choice(images, size = 1)[0]

    if index >= len(images):
        return None

    return images[index]

def apply_if_not_none(fn, x):
    if x is None:
        return None

    return fn(x)

def load_images(paths, transform = None):
    result = [ apply_if_not_none(default_loader, x) for x in paths ]

    if transform is not None:
        result = [ apply_if_not_none(transform, x) for x in result ]

    return result

