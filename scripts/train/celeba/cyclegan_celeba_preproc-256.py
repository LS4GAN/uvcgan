import argparse
import os

from uvcgan import ROOT_OUTDIR, train, join_dicts
from uvcgan.utils.parsers import add_preset_name_parser

def parse_cmdargs():
    parser = argparse.ArgumentParser(
        description = 'Train CelebA CycleGANs'
    )

    add_preset_name_parser(parser, 'gen',   GEN_PRESETS,      'vit-unet-12')
    add_preset_name_parser(parser, 'cycle', CYCLEGAN_PRESETS, 'cycle_mid')
    add_preset_name_parser(parser, 'gp',    GP_PRESETS,       'paper')

    add_preset_name_parser(parser, 'loss',     [ 'lsgan', 'wgan' ],  'lsgan')
    add_preset_name_parser(
        parser, 'transfer', [ 'none', 'imagenet', 'self' ], 'self'
    )

    add_preset_name_parser(
        parser, 'attr',  [ 'male2female', 'glasses' ], 'male2female'
    )

    return parser.parse_args()

def get_transfer_preset(cmdargs):
    if (cmdargs.transfer is None) or (cmdargs.transfer == 'none'):
        return None

    if cmdargs.transfer == 'imagenet':
        base_model = (
            'bert_imagenet/model_d(imagedir)_m(simple-autoencoder)_d(None)'
            f"_g({GEN_PRESETS[cmdargs.gen]['model']})_bert-{cmdargs.gen}-256"
        )

    if cmdargs.transfer == 'self':
        base_model = (
            'celeba_preproc/model_d(imagedir)_m(simple-autoencoder)_d(None)'
            f"_g({GEN_PRESETS[cmdargs.gen]['model']})_bert-{cmdargs.gen}-256"
        )

    return {
        'base_model' : base_model,
        'transfer_map'  : {
            'gen_ab' : 'encoder',
            'gen_ba' : 'encoder',
        },
        'strict'        : True,
        'allow_partial' : False,
    }

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

GP_PRESETS = {
    'none'   : None,
    'common' : { 'lambda_gp' : 10 },
    # see https://arxiv.org/pdf/1710.10196.pdf
    'low'    : { 'lambda_gp' : 1/10**2,    'constant' : 10 },
    'paper'  : { 'lambda_gp' : 0.1/100**2, 'constant' : 100 },
    'large'  : { 'lambda_gp' : 1/750**2,   'constant' : 750 },
}

CYCLEGAN_PRESETS = {
    'cycle_high' : {
        'lambda_a'   : 10.0,
        'lambda_b'   : 10.0,
        'lambda_idt' : 0.5,
    },
    'cycle_high_noidt' : {
        'lambda_a'   : 10.0,
        'lambda_b'   : 10.0,
        'lambda_idt' : 0.0,
    },
    'cycle_mid' : {
        'lambda_a'   : 5.0,
        'lambda_b'   : 5.0,
        'lambda_idt' : 0.5,
    },
    'cycle_mid_noidt' : {
        'lambda_a'   : 5.0,
        'lambda_b'   : 5.0,
        'lambda_idt' : 0.0,
    },
    'cycle_low' : {
        'lambda_a'   : 1.0,
        'lambda_b'   : 1.0,
        'lambda_idt' : 0.5,
    },
    'cycle_low_noidt' : {
        'lambda_a'   : 1.0,
        'lambda_b'   : 1.0,
        'lambda_idt' : 0.0,
    },
}

cmdargs   = parse_cmdargs()
args_dict = {
    'batch_size' : 1,
    'data' : {
        'dataset'      : 'cyclegan',
        'dataset_args' : {
            'path'        : f'celeba_{cmdargs.attr}',
            'align_train' : False,
        },
        'transform_train' : [
            'random-flip-horizontal',
            { 'name' : 'resize',      'size' : 256, },
            { 'name' : 'random-crop', 'size' : 256, },
        ],
        'transform_val' : [
            { 'name' : 'resize',      'size' : 256, },
            { 'name' : 'center-crop', 'size' : 256, },
        ],
    },
    'image_shape' : (3, 256, 256),
    'epochs'      : 500,
    'discriminator' : {
        'model' : 'basic',
        'model_args' : None,
        'optimizer'  : {
            'name'  : 'Adam',
            'lr'    : 1e-4,
            'betas' : (0.5, 0.99),
        },
        'weight_init' : {
            'name'      : 'normal',
            'init_gain' : 0.02,
        },
    },
    'generator' : {
        **GEN_PRESETS[cmdargs.gen],
        'optimizer'  : {
            'name'  : 'Adam',
            'lr'    : 1e-4,
            'betas' : (0.5, 0.99),
        },
        'weight_init' : {
            'name'      : 'normal',
            'init_gain' : 0.02,
        },
    },
    'model' : 'cyclegan',
    'model_args' : {
        **CYCLEGAN_PRESETS[cmdargs.cycle],
        'pool_size'  : 50,
    },
    'scheduler' : {
        'name'          : 'linear',
        'epochs_warmup' : 250,
        'epochs_anneal' : 250,
    },
    'loss'             : cmdargs.loss,
    'gradient_penalty' : GP_PRESETS[cmdargs.gp],
    'steps_per_epoch'  : 2000,
    'transfer'         : get_transfer_preset(cmdargs),
# args
    'label'  : (
        f'cyclegan_{cmdargs.gen}_{cmdargs.attr}_{cmdargs.transfer}'
        f'-{cmdargs.loss}-{cmdargs.gp}-{cmdargs.cycle}-256'
    ),
    'outdir' : os.path.join(ROOT_OUTDIR, 'celeba_preproc'),
    'log_level'  : 'DEBUG',
    'checkpoint' : 50,
    'workers'    : 1,       # for reproducibility
}

train(args_dict)

