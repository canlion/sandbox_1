import datetime
import tensorflow as tf

from yolo_config import CFG
from model import YoloV1
from loss import YoloV1Loss
from data import YoloData

from config import dict2namedtuple
from data_load.data_loader import DataLoader
from optimizer import get_optimizer
from learning_rate import get_learning_rate


# parameter namedtuple
params = dict2namedtuple(CFG)
mp = params.train.mixed_precision

if mp:
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
    tf.keras.mixed_precision.experimental.set_policy(policy)
    print('compute : {}, variables : {}'.format(policy.compute_dtype, policy.variable_dtype))
    tf.config.optimizer.set_jit(True)


# setting
model = YoloV1(params)

data_loader = DataLoader(params)
yolo_data = YoloData(params)
train_ds = data_loader.set_ds(yolo_data.map_fn_train, training=True)
eval_ds = data_loader.set_ds(yolo_data.map_fn_eval, training=False)

lr = get_learning_rate(params)
optimizer = get_optimizer(lr, params)

if mp:
    optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, "dynamic")

loss_obj = YoloV1Loss(params)

# metric
loss_mean_train = tf.keras.metrics.Mean(name='loss_train')
loss_mean_eval = tf.keras.metrics.Mean(name='loss_eval')

# tensorboard
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir_path = './tensorboard/' + current_time + '/{}'
log_dir_train = log_dir_path.format('train')
log_dir_eval = log_dir_path.format('eval')
summary_writer_train = tf.summary.create_file_writer(log_dir_train)
summary_writer_eval = tf.summary.create_file_writer(log_dir_eval)


@tf.function
def train(x, y):
    with tf.GradientTape() as tape:
        pred = model(x, training=True)
        loss = loss_obj(y, pred)
        loss_l2 = loss + tf.add_n(model.losses)
    gradients = tape.gradient(loss_l2, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    loss_mean_train.update_state(loss)
    return loss


@tf.function
def train_network_mp(x, y):
    with tf.GradientTape() as tape:
        pred = model(tf.cast(x, tf.float16), training=True)
        loss = loss_obj(y, tf.cast(pred, tf.float32))
        loss_l2 = loss + tf.math.add_n(model.losses)
        scaled_loss = optimizer.get_scaled_loss(loss_l2)
    scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
    gradients = optimizer.get_unscaled_gradients(scaled_gradients)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    loss_mean_train.update_state(loss)
    return loss


@tf.function
def eval(x, y, mp=False):
    dtype = tf.float16 if mp else tf.float32
    pred = model(tf.cast(x, dtype), training=False)
    loss = loss_obj(y, tf.cast(pred, tf.float32))

    loss_mean_eval.update_state(loss)


if __name__ == '__main__':
    train_report_format = '[{:3}, {:6}] mean loss : {} / lr : {:.6f} / ignored step : {}'
    report_format = '[----- evaluation {:3} -----] mean loss : {:.3f}'
    train_fn = train_network_mp if mp else train
    ignored_step = 0

    for step, (x, y) in enumerate(train_ds, 0):
        t_loss = train_fn(x, y)

        if (step+1) % (params.train.eval_step//10) == 0:
            opt_step = tf.cast(optimizer._optimizer._iterations if mp else optimizer._iterations, tf.float32)
            ignored_step = (step+ - opt_step)
            print(train_report_format.format(step//params.train.eval_step + 1,
                                             step + 1,
                                             loss_mean_train.result(),
                                             optimizer.lr(opt_step),
                                             ignored_step))
            with summary_writer_train.as_default():
                tf.summary.scalar('loss', loss_mean_train.result(), step=step)
            loss_mean_train.reset_states()

        if step and (step+1) % params.train.eval_step == 0:
            for x, y in eval_ds:
                eval(x, y)

            print(report_format.format(step//params.train.eval_step + 1, loss_mean_eval.result()))
            with summary_writer_eval.as_default():
                tf.summary.scalar('loss', loss_mean_eval.result(), step=step)
            loss_mean_train.reset_states()
            loss_mean_eval.reset_states()

            model.save_weights('./model_save/yolo_v1_{}'.format(step))

        if step >= params.train.total_step:
            break
