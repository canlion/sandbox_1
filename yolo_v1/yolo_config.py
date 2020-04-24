CFG = {
    'data_loader': {
        'dataset': 'voc',
        'train_tfr_pattern': '/sandbox_datasets/voc_download/tfrecord/train/*',
        'eval_tfr_pattern': '/sandbox_datasets/voc_download/tfrecord/val/*',
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
            'l2_decay': 5e-4,
            'lambda_coord': 5.,
            'lambda_noobj': .5,
        }
    },
    'train': {
        'mixed_precision': True,
        'total_step': 120000,
        'eval_step': 2000,
        'batch_size': 8,
        'learning_rate': {
            'schedule': 'PiecewiseConstantDecay',
            'boundaries': [60000, 90000],
            'values': [1e-4, 1e-5, 1e-6],
            'warmup_learning_rate': 0.,
            'warmup_steps': 2000,
        },
        'optimizer': {
            'policy': 'SGD',
            'momentum': .9,
            'nesterov': False,
        }
    }
}
