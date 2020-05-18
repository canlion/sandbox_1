CFG = {
    'data_loader': {
        'dataset': 'voc',
        'train_tfr_pattern': '/sandbox/datasets/voc/train/*',
        'eval_tfr_pattern': '/sandbox/datasets/voc/val/*',
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
        'mixed_precision': False,
        'total_step': 240000,
        'eval_step': 2000,
        'train_batch_size': 8,
        'learning_rate': {
            'schedule': 'PiecewiseConstantDecay',
            'boundaries': [12000, 18000],
            'values': [1e-4, 1e-5, 1e-6],
            'warmup_learning_rate': 1e-5,
            'warmup_steps': 2000,
        },
        'optimizer': {
            'policy': 'Adam',
        }
    }
}
