from collections import OrderedDict

CFG = OrderedDict({
    'data_loader': {
        'dataset': 'voc',
        'train_tfr_pattern': '/mnt/hdd/jinwoo/sandbox_datasets/voc_download/tfrecord/train/*',
        'eval_tfr_pattern': '/mnt/hdd/jinwoo/sandbox_datasets/voc_download/tfrecord/val/*',
    },
    'network': {
        'input_size': [448, 448, 3],
        'negative_slope': .1,
        'backbone': {
            'architecture': 'resnet',
            'depth': 50,
            'l2_decay': 5e-4,
        },
        'yolo': {
            'S': 7,
            'B': 2,
            'classes': 20,
            'l2_decay': 5e-5,
            'lambda_coord': 5.,
            'lambda_noobj': .5,
        }
    },
    'train': {
        'mixed_precision': True,
        'total_step': 480000,
        'eval_step': 4000,
        'batch_size': 4,
        'learning_rate': {
            'schedule': 'PiecewiseConstantDecay',
            'boundaries': [24000, 36000],
            'values': [1e-4, 1e-5, 1e-6],
            'warmup_learning_rate': 1e-5,
            'warmup_steps': 4000,
        },
        'optimizer': {
            'policy': 'Adam',
        }
    }
})
