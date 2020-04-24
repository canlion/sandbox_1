import tensorflow as tf


def get_optimizer(lr, params):
    params_dict = params.train.optimizer._asdict()
    optimizer = eval('tf.keras.optimizers.{}'.format(params_dict.pop('policy')))

    return optimizer(lr, **params_dict)
