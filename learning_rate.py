import tensorflow as tf
from tensorflow.keras.optimizers.schedules import *


def get_learning_rate(params):
    params_dict = params.train.learning_rate._asdict()
    policy = params_dict.pop('schedule')
    schedule = eval('policy')

    if params_dict.get('warmup_steps'):
        warmup_lr = params_dict.pop('warmup_learning_rate')
        warmup_steps = params_dict.pop('warmup_steps')
        return LRWarmup(schedule(**params_dict), warmup_steps, warmup_lr)
    else:
        return schedule(**params_dict)


class LRWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, scheduler, step, init_lr=0):
        super(LRWarmup, self).__init__()
        assert step > 0, 'learning rate warm-up step must be larger than zero.'

        self.init_lr = init_lr
        self.scheduler = scheduler
        self.lr = self.scheduler(0)
        self.step = step

    @tf.function
    def __call__(self, steps):
        if tf.less_equal(steps, self.step):
            return self.init_lr + (self.lr-self.init_lr) * (steps / self.step)
        else:
            return self.scheduler(steps - self.step)
