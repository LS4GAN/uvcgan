# Overview

This package provides CycleGAN and generator implementations used in the
`uvcgan` [paper][uvcgan_paper].

`uvcgan` introduces an improved method to perform an unpaired image-to-image
style transfer based on a CycleGAN framework. Combined with a new hybrid
generator architecture UNet-ViT (UNet-Vision Transformer) and a self-supervised
pre-training, it achieves state-of-the-art results on a multitude of style
transfer benchmarks.

This README file provides brief instructions about how to set up the `uvcgan`
package and reproduce the results of the paper.

The accompanying [benchmarking][benchmarking_repo] repository contains detailed
instructions on how competing CycleGAN, CouncilGAN, ACL-GAN, and U-GAT-IT
models were trained and evaluated.

![benchmark_grid](https://user-images.githubusercontent.com/22546248/156432283-39390ec5-28a0-41d9-8674-b7d15a46e692.jpg)

# Installation & Requirements

## Requirements

`uvcgan` was trained using the official `pytorch` container
`pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime`. You can setup a similar
training environment with `conda`
```
conda env create -f contrib/conda_env.yml
```

## Installation

To install the `uvcgan` package one may simply run the following command
```
python setup.py develop --user
```
from the `uvcgan` source tree.

## Environment Setup

`uvcgan` extensively uses two environment variables `UVCGAN_DATA` and
`UVCGAN_OUTDIR` to locate user data and output directories. Users are advised
to set these environment variables. `uvcgan` will look for datasets in the
`${UVCGAN_DATA}` directory and will save results under the `"${UVCGAN_OUTDIR}"`
directory. If these variables are not set then they will default to `./data`
and `./outdir` correspondingly.


# UVCGAN Reproduction

To reproduce the results of the `uvcgan` paper, the following workflow is
suggested:

1. Download CycleGAN datasets (`selfie2anime`, `celeba`).
2. Pre-train generators in a BERT-like setup.
3. Train CycleGAN models.
4. Generate translated images and evaluate KID/FID scores.

We also provide pre-trained generators that were used to obtain the `uvcgan`
[paper][uvcgan_paper] results. They can be found [here][pretrained_models].


## 1. Download CycleGAN Datasets

`uvcgan` provides a script (`scripts/download_dataset.sh`) to download and
unpack various CycleGAN datasets.

For example, one can use the following commands to download `selfie2anime`,
CelebA `male2female`, CelebA `eyeglasses`, and the un-partitioned CelebA
datasets:

```bash
./scripts/download_dataset.sh selfie2anime
./scripts/download_dataset.sh male2female
./scripts/download_dataset.sh glasses
./scripts/download_dataset.sh celeba_all
```

If you want to pre-train generators on the `ImageNet` dataset, a manual
download of this dataset is required. More details about the origins of these
datasets can be found [here](doc/datasets.md).


## 2. Pre-training Generators

To pre-train CycleGAN generators in a BERT-like setup one can use the
following three scripts:
```
scripts/train/anime2selfie/bert_anime2selfie-256.py
scripts/train/bert_imagenet/bert_imagenet-256.py
scripts/train/celeba/bert_celeba_preproc-256.py
```

All three scripts have similar invocation. For example, to pre-train generators
on the `selfie2anime` dataset one can run:
```
python scripts/train/anime2selfie/bert_anime2selfie-256.py
```
You can find more details by looking over the scripts, which contain training
configuration and are rather self-explanatory.

The pre-trained generators will be saved under the "${UVCGAN_OUTDIR}"
directory.


## 3. Training CycleGAN Generators

Similarly to the generator pre-training, `uvcgan` provides two scripts to
train the CycleGAN models:
```
scripts/train/anime2selfie/cyclegan_anime2selfie-256.py
scripts/train/celeba/cyclegan_celeba_preproc-256.py
```

Their invocation is similar to the corresponding scripts of the generator
pre-training scripts. For example, the following command will train the
CycleGAN model to perform male-to-female transfer

```bash
python scripts/train/celeba/cyclegan_celeba_preproc-256.py --attr male2female
```

More details can be found by looking over these scripts. The trained CycleGAN
models will be saved under the "${UVCGAN_OUTDIR}" directory.


## 4. Evaluation of the trained model

To perform the style transfer with the trained models `uvcgan` provides a
script `scripts/translate_images.py`. Its invocation is simple
```
python scripts/translate_images.py PATH_TO_TRAINED_MODEL -n 100
```
where `-n` parameter controls the number of images from the test dataset to
translate. The original and translated images will be saved under
`PATH_TO_TRAINED_MODEL/evals/final/translated`

You can use the [torch_fidelity](https://github.com/toshas/torch-fidelity)
package to evaluate KID/FID metrics on the translated images. Please, refer to
the accompanying [benchmarking][benchmarking_repo] repository for the KID/FID
evaluation details.


# F.A.Q.

## I am training my model on a multi-GPU node. How to make sure that I use only one GPU?

You can specify GPUs that `pytorch` will use with the help of the
`CUDA_VISIBLE_DEVICES` environment variable. This variable can be set to a list
of comma-separated GPU indices. When it is set, `pytorch` will only use GPUs
whose IDs are in the `CUDA_VISIBLE_DEVICES`.


# Contributing

All contributions are welcome. To ensure code consistency among a diverse set
of collaborators, `uvcgan` uses `pylint` linter that automatically identifies
common code issues and ensures uniform code style.

If you are submitting code changes, please run the `pylint` tool over your code
and verify that there are no issues.

# LICENSE

`uvcgan` is distributed under `BSD-2` license.

`uvcgan` repository contains some code (primarity in `uvcgan/base`
subdirectory) from [pytorch-CycleGAN-and-pix2pix][cyclegan_repo].
This code is also licensed under `BSD-2` license (please refer to
`uvcgan/base/LICENSE` for details). Each code snippet that was taken from
[pytorch-CycleGAN-and-pix2pix][cyclegan_repo] has a note about proper
copyright attribution.


[cyclegan_repo]: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
[benchmarking_repo]: https://github.com/LS4GAN/benchmarking
[uvcgan_paper]: https://arxiv.org/abs/2203.02557
[pretrained_models]: https://zenodo.org/record/6336010

