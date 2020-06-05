import tensorflow as tf
from generate_network import backbone_generator


class YoloV1(tf.keras.Model):
    def __init__(self, params):
        super(YoloV1, self).__init__()
        self.input_size = params.network.input_size
        self.backbone = backbone_generator(params)
        self.preprocess_input = self.backbone.preprocess_input

        self.S = params.network.yolo.S
        self.B = params.network.yolo.B
        self.classes = params.network.yolo.classes
        self.lambda_coord = params.network.yolo.lambda_coord
        self.lambda_noobj = params.network.yolo.lambda_noobj

        self.regularizer = tf.keras.regularizers.l2(params.network.yolo.l2_decay)
        self.negative_slope = params.network.negative_slope
        self.net = None
        self.build_network()

    def build_network(self):
        self.net = tf.keras.Sequential([
            self.backbone.network,
            tf.keras.layers.Conv2D(512, 3, 1, 'same', kernel_regularizer=self.regularizer),
            tf.keras.layers.ReLU(negative_slope=self.negative_slope),
            tf.keras.layers.Conv2D(512, 3, 1, 'same', kernel_regularizer=self.regularizer),
            tf.keras.layers.ReLU(negative_slope=self.negative_slope),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024, kernel_regularizer=self.regularizer),
            tf.keras.layers.ReLU(negative_slope=self.negative_slope),
            tf.keras.layers.Dropout(.5),
            tf.keras.layers.Dense(self.S*self.S*(self.B*5+self.classes), kernel_regularizer=self.regularizer),
            tf.keras.layers.Reshape((self.S, self.S, (self.B*5+self.classes))),
            # ResidualBlock(256, 2, self.regularizer, True, self.negative_slope),
            # ResidualBlock(256, 1, self.regularizer, True, self.negative_slope),
            # ResidualBlock(128, 1, self.regularizer, True, self.negative_slope),
            # tf.keras.layers.Conv2D(self.B * 5 + self.classes, 1, 1, 'same',
            #                        kernel_regularizer=self.regularizer,
            #                        kernel_initializer=tf.keras.initializers.he_normal(),
            #                        activation='linear')
        ])

    def call(self, inputs, training=False):
        # inputs = self.preprocess_input(inputs)
        inputs = (inputs / 127.5) - 1
        return self.net(inputs, training)


class ResidualBlock(tf.keras.Model):
    def __init__(self, filters, strides, regularizer=None, match_dims=False, negative_slope=0.):
        super(ResidualBlock, self).__init__()
        self.match_dims = strides != 1 or match_dims
        self.negative_slope = negative_slope

        if self.match_dims:
            self.seq_shortcut = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters * 4, 1, strides, 'same', kernel_regularizer=regularizer,
                                       kernel_initializer=tf.keras.initializers.he_normal()),
                tf.keras.layers.BatchNormalization()
            ])

        self.seq = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, 1, strides, 'same', kernel_regularizer=regularizer,
                                   kernel_initializer=tf.keras.initializers.he_normal()),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(negative_slope=self.negative_slope),
            tf.keras.layers.Conv2D(filters, 3, 1, 'same', kernel_regularizer=regularizer,
                                   kernel_initializer=tf.keras.initializers.he_normal()),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(negative_slope=self.negative_slope),
            tf.keras.layers.Conv2D(filters * 4, 1, 1, 'same', kernel_regularizer=regularizer,
                                   kernel_initializer=tf.keras.initializers.he_normal()),
            tf.keras.layers.BatchNormalization(),
        ])

    def call(self, inputs, training=False):
        shortcut = inputs
        if self.match_dims:
            shortcut = self.seq_shortcut(shortcut, training)
        x = self.seq(inputs, training)
        return tf.keras.layers.ReLU(negative_slope=self.negative_slope)(x + shortcut)
