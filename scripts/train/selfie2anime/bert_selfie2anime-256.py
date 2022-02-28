import argparse
import os

from uvcgan import ROOT_OUTDIR, train
from uvcgan.utils.parsers import add_preset_name_parser, add_batch_size_parser

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Train Anime2Selfie BERT')
    add_preset_name_parser(parser, 'gen', GEN_PRESETS, 'vit-unet-12')
    add_batch_size_parser(parser, default = 64)
    return parser.parse_args()

GEN_PRESETS = {
    'resnet9' : {
        'model'      : 'resnet_9blocks',
        'model_args' : None,
    },
    'unet' : {
        'model'      : 'unet_256',
        'model_args' : None,
    },
    'vit-unet-6' : {
        'model' : 'vit-unet',
        'model_args' : {
            'features'           : 384,
            'n_heads'            : 6,
            'n_blocks'           : 6,
            'ffn_features'       : 1536,
            'embed_features'     : 384,
            'activ'              : 'gelu',
            'norm'               : 'layer',
            'unet_features_list' : [48, 96, 192, 384],
            'unet_activ'         : 'leakyrelu',
            'unet_norm'          : 'instance',
            'unet_downsample'    : 'conv',
            'unet_upsample'      : 'upsample-conv',
            'rezero'             : True,
            'activ_output'       : 'sigmoid',
        },
    },
    'vit-unet-12' : {
        'model' : 'vit-unet',
        'model_args' : {
            'features'           : 384,
            'n_heads'            : 6,
            'n_blocks'           : 12,
            'ffn_features'       : 1536,
            'embed_features'     : 384,
            'activ'              : 'gelu',
            'norm'               : 'layer',
            'unet_features_list' : [48, 96, 192, 384],
            'unet_activ'         : 'leakyrelu',
            'unet_norm'          : 'instance',
            'unet_downsample'    : 'conv',
            'unet_upsample'      : 'upsample-conv',
            'rezero'             : True,
            'activ_output'       : 'sigmoid',
        },
    },
}

cmdargs   = parse_cmdargs()
args_dict = {
    'batch_size' : cmdargs.batch_size,
    'data' : {
        'dataset' : 'cyclegan',
        'dataset_args'   : {
            'path'        : 'selfie2anime',
            'align_train' : False,
        },
        'transform_train' : [
            { 'name' : 'resize',          'size'    : 286, },
            { 'name' : 'random-rotation', 'degrees' : 10,  },
            { 'name' : 'random-crop',     'size'    : 256, },
            'random-flip-horizontal',
            {
                'name' : 'color-jitter',
                'brightness' : 0.2,
                'contrast'   : 0.2,
                'saturation' : 0.2,
                'hue'        : 0.2,
            },
        ],
    },
    'image_shape' : (3, 256, 256),
    'epochs'      : 2499,
    'discriminator' : None,
    'generator' : {
        **GEN_PRESETS[cmdargs.gen],
        'optimizer'  : {
            'name'  : 'AdamW',
            'lr'    : cmdargs.batch_size * 5e-3 / 512,
            'betas' : (0.9, 0.99),
            'weight_decay' : 0.05,
        },
        'weight_init' : {
            'name'      : 'normal',
            'init_gain' : 0.02,
        }
    },
    'model'      : 'autoencoder',
    'model_args' : {
        'joint'   : True,
        'masking' : {
            'name'       : 'image-patch-random',
            'patch_size' : (32, 32),
            'fraction'   : 0.4,
        },
    },
    'scheduler' : {
        'name'      : 'CosineAnnealingWarmRestarts',
        'T_0'       : 500,
        'T_mult'    : 1,
        'eta_min'   : cmdargs.batch_size * 5e-8 / 512,
    },
    'loss'             : 'l1',
    'gradient_penalty' : None,
    'steps_per_epoch'  : 32 * 1024 // cmdargs.batch_size,
# args
    'label'      : f'bert-{cmdargs.gen}-256',
    'outdir'     : os.path.join(ROOT_OUTDIR, 'selfie2anime'),
    'log_level'  : 'DEBUG',
    'checkpoint' : 100,
}

train(args_dict)

