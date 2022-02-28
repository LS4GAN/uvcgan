import os
from speval import speval
from cyclegan import ROOT_OUTDIR, train, join_dicts

args_dict = {
    'batch_size' : 1024,
    'data' : 'toyzero-presimple',
    'data_args'   : {
        'path'     : 'toyzero-1k',
        'fname'    : 'test_1_n100-U-128x128.csv',
        'shuffle'  : False,
        'val_size' : 1000,
    },
    'image_shape' : (1, 128, 128),
    'epochs'      : 300,
    'discriminator' : { 'model' : None, },
    'generator' : {
        'model' : 'vit-v0',
        'model_args' : {
            'features'       : 768,
            'n_heads'        : 12,
            'n_blocks'       : 12,
            'ffn_features'   : 3072,
            'embed_features' : 768,
            'activ'          : 'gelu',
            'norm'           : 'layer',
            'token_size'     : (16, 16),
            'rescale'        : False,
            'rezero'         : True,
        },
        'optimizer'  : {
            'name'  : 'AdamW',
            'lr'    : 1e-3,
            'betas' : (0.9, 0.99),
            'weight_decay' : 0.05,
        },
        'weight_init' : {
            'name'      : 'normal',
            'init_gain' : 0.02,
        },
    },
    'model'      : 'nogan',
    'model_args' : None,
    'scheduler' : {
        'name'      : 'cosine',
        'T_max'     : 10,
        'eta_min'   : 1e-6,
    },
    'loss'             : 'wgan',
    'gradient_penalty' : { 'lambda_gp' : 1 },
    'steps_per_epoch'  : 10000,
# args
    'label'  : None,
    'outdir' : os.path.join(
        ROOT_OUTDIR, 'experiments', 'toyzero-128-vit-nogan-test-v0'
    ),
    'log_level'  : 'DEBUG',
    'checkpoint' : 1,
}

train(args_dict)

