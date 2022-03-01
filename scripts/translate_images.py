#!/usr/bin/env python

import argparse
import collections
import os

import tqdm
import numpy as np
from PIL import Image

from uvcgan.eval.funcs import (
    load_eval_model_dset_from_cmdargs, tensor_to_image, slice_data_loader
)
from uvcgan.utils.parsers import (
    add_standard_eval_parsers, add_plot_extension_parser
)

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Translate images')

    add_standard_eval_parsers(parser, default_n_eval = None)
    add_plot_extension_parser(parser, default = ('png', ))

    return parser.parse_args()

def plot_model_images(plotdir, sample_counter, model, ext):
    for (name, torch_image) in model.images.items():
        if torch_image is None:
            continue

        # pylint: disable=consider-using-enumerate
        for index in range(len(torch_image)):
            sample_idx = sample_counter[name]
            fname = f'sample_{sample_idx}'
            path  = os.path.join(plotdir, name, fname)

            image = tensor_to_image(torch_image[index])
            image = (255 * image).astype(np.uint8)
            image = Image.fromarray(image)

            for e in ext:
                image.save(path + '.' + e)

            sample_counter[name] += 1

def plot_images(model, it_val, n_eval, batch_size, plotdir, ext):
    # pylint: disable=too-many-arguments
    it_val, steps = slice_data_loader(it_val, batch_size, n_eval)

    for name in model.images:
        os.makedirs(os.path.join(plotdir, name), exist_ok = True)

    sample_counter = collections.defaultdict(int)

    for batch in tqdm.tqdm(it_val, desc = 'Plotting', total = steps):
        model.set_input(batch)
        model.forward_nograd()

        plot_model_images(plotdir, sample_counter, model, ext)

def get_plotdir(evaldir, model_state):
    return os.path.join(evaldir, f'translated_images_{model_state}')

def main():
    cmdargs = parse_cmdargs()
    args, model, it_val, evaldir = load_eval_model_dset_from_cmdargs(cmdargs)

    plotdir = get_plotdir(evaldir, cmdargs.model_state)
    plot_images(
        model, it_val, cmdargs.n_eval, args.batch_size, plotdir, cmdargs.ext
    )

if __name__ == '__main__':
    main()

