import tensorflow as tf
from .tfr_parser import voc_parser

class DataLoader:
    def __init__(self, params):
        train_tfrs = tf.data.TFRecordDataset.list_files(params.data_loader.train_tfr_pattern, shuffle=True)
        eval_tfrs = tf.data.TFRecordDataset.list_files(params.data_loader.eval_tfr_pattern, shuffle=False)
        self.train_dataset = tf.data.TFRecordDataset(train_tfrs)
        self.eval_dataset = tf.data.TFRecordDataset(eval_tfrs)
        self.batch_size = params.train.batch_size

        if params.data_loader.dataset == 'voc':
            self.parser = voc_parser
        else:
            raise Exception('not supported dataset.')

    def set_ds(self, map_fn = None, training=True):
        if training:
            ds = self.train_dataset.map(self.parser, tf.data.experimental.AUTOTUNE)
            ds = ds.repeat().shuffle(1000)
        else:
            ds = self.eval_dataset.map(self.parser, tf.data.experimental.AUTOTUNE)

        if map_fn:
            ds = ds.map(map_fn, tf.data.experimental.AUTOTUNE)\
                .batch(self.batch_size, drop_remainder=True)\
                .prefetch(tf.data.experimental.AUTOTUNE)

        return ds