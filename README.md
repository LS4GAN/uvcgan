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

For anyone interested in applying `uvcgan` over a scientific dataset, we
publish a tutorial/demonstration of applying the `uvcgan` over the neutrino
data at [uvcgan4slats](https://github.com/LS4GAN/uvcgan4slats).

![benchmark_grid](https://user-images.githubusercontent.com/22546248/156432283-39390ec5-28a0-41d9-8674-b7d15a46e692.jpg)

NOTE: The default cyclegan dataset implementation automatically converts
grayscale images into RGB. If you like to apply `uvcgan` to a grayscale
dataset, consider replacing the `cyclegan` dataset implementation with a
`cyclegan-v2` (introduced in d54411c79a0ce49a74ecb48b41a7bb11ffe2b385).


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

**NOTE**: As of June 2023, the CelebA datasets (`male2female` and `glasses`)
need to be recreated manually. Please refer to
[celeba4cyclegan](https://github.com/LS4GAN/celeba4cyclegan) for instructions
on how to do that.

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
scripts/train/selfie2anime/bert_selfie2anime-256.py
scripts/train/bert_imagenet/bert_imagenet-256.py
scripts/train/celeba/bert_celeba_preproc-256.py
```

All three scripts have similar invocation. For example, to pre-train generators
on the `selfie2anime` dataset one can run:
```
python scripts/train/selfie2anime/bert_selfie2anime-256.py
```
You can find more details by looking over the scripts, which contain training
configuration and are rather self-explanatory.

The pre-trained generators will be saved under the "${UVCGAN_OUTDIR}"
directory.


## 3. Training CycleGAN Generators

Similarly to the generator pre-training, `uvcgan` provides two scripts to
train the CycleGAN models:
```
scripts/train/selfie2anime/cyclegan_selfie2anime-256.py
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


# Additional Examples

The additional usage examples can be found in the `examples` subdirectory of
the `uvcgan` package.


# F.A.Q.

## I am training my model on a multi-GPU node. How to make sure that I use only one GPU?

You can specify GPUs that `pytorch` will use with the help of the
`CUDA_VISIBLE_DEVICES` environment variable. This variable can be set to a list
of comma-separated GPU indices. When it is set, `pytorch` will only use GPUs
whose IDs are in the `CUDA_VISIBLE_DEVICES`.


## What is the structure of a model directory?

`uvcgan` saves each model in a separate directory that contains:
 - `MODEL/config.json` -- model architecture, training, and evaluation
    configurations
 - `MODEL/net_*.pth`  -- PyTorch weights of model networks
 - `MODEL/opt_*.pth`  -- PyTorch weights of training optimizers
 - `MODEL/shed_*.pth` -- PyTorch weights of training schedulers
 - `MODEL/checkpoints/` -- training checkpoints
 - `MODEL/evals/`     -- evaluation results


## Training fails with "Config collision detected" error

`uvcgan` enforces a one-model-per-directory policy to prevent accidental
overwrites of existing models. Each model directory must have a unique
configuration - if you try to place a model with different settings in a
directory that already contains a model, you'll receive a "Config collision
detected" error.

This safeguard helps prevent situations where you might accidentally lose
trained models by starting a new training run with different parameters in the
same directory.

Solutions:
1. To overwrite the old model: delete the old `config.json` configuration file
   and restart the training process.
2. To preserve the old model: modify the training script of the new model and
   update the `label` or `outdir` configuration options to avoid collisions.


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

# Citation
If you use this code for your research, please cite our paper.
```
@inproceedings{torbunov2023uvcgan,
  title     = {Uvcgan: Unet vision transformer cycle-consistent gan for unpaired image-to-image translation},
  author    = {Torbunov, Dmitrii and Huang, Yi and Yu, Haiwang and Huang, Jin and Yoo, Shinjae and Lin, Meifeng and Viren, Brett and Ren, Yihui},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages     = {702--712},
  year      = {2023}
}
```

[cyclegan_repo]: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
[benchmarking_repo]: https://github.com/LS4GAN/benchmarking
[uvcgan_paper]: https://arxiv.org/abs/2203.02557
[pretrained_models]: https://zenodo.org/record/6336010

