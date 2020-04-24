import tensorflow as tf


def voc_parser(example):
    feature_description = {
        'folder': tf.io.FixedLenFeature([], tf.string),
        'filename': tf.io.FixedLenFeature([], tf.string),
        'image': tf.io.FixedLenFeature([], tf.string),
        'size': tf.io.FixedLenFeature([3], tf.int64),
        'label': tf.io.VarLenFeature(tf.int64),
        'truncated': tf.io.VarLenFeature(tf.int64),
        'difficult': tf.io.VarLenFeature(tf.int64),
        'xmin': tf.io.VarLenFeature(tf.float32),
        'ymin': tf.io.VarLenFeature(tf.float32),
        'xmax': tf.io.VarLenFeature(tf.float32),
        'ymax': tf.io.VarLenFeature(tf.float32),
    }

    features = tf.io.parse_single_example(example, feature_description)
    update_dict = dict()
    for key, val in features.items():
        if isinstance(val, tf.SparseTensor):
            update_dict[key] = tf.sparse.to_dense(val)

    update_dict['image'] = tf.io.decode_jpeg(features['image'])

    features.update(update_dict)
    return features