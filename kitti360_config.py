import os.path as osp

WINDOWS = {
    'train': [],

    'val': [
        '2013_05_28_drive_0000_sync/0000000372_0000000610',
    ],

    'test': []
}

SEQUENCES = {
    k: list(set(osp.dirname(x) for x in v)) for k, v in WINDOWS.items()}

CACHING_ENABLED = True
LOADING_ENABLED = False
