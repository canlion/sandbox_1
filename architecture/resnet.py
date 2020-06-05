import tensorflow as tf
import tensorflow.keras as K


class ResNet:
    def __init__(self, params):
        # super(ResNet, self).__init__()

        model_config = {
            10: {'block': self.residual_block, 'layers': [1, 1, 1, 1]},
            18: {'block': self.residual_block, 'layers': [2, 2, 2, 2]},
            34: {'block': self.residual_block, 'layers': [3, 4, 6, 3]},
            50: {'block': self.bottleneck_block, 'layers': [3, 4, 6, 3]},
            101: {'block': self.bottleneck_block, 'layers': [3, 4, 23, 3]},
            152: {'block': self.bottleneck_block, 'layers': [3, 8, 36, 3]},
            200: {'block': self.bottleneck_block, 'layers': [3, 24, 36, 3]}
        }

        model_weights = {
            50: '/mnt/hdd/jinwoo/sandbox_datasets/pre_trained/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        }

        architecture = model_config[params.network.backbone.depth]
        self.block = architecture['block']
        self.n_list = architecture['layers']
        self.input_size = params.network.input_size
        self.regularizer = tf.keras.regularizers.l2(params.network.backbone.l2_decay)
        self.negative_slope = params.network.negative_slope

        self.network, self.intermediate = self.generate_network()
        self.network.load_weights(model_weights[params.network.backbone.depth])
        self.preprocess_input = tf.keras.applications.resnet.preprocess_input

    def generate_network(self):
        inputs = K.layers.Input(self.input_size, name='input_resnet')
        x = K.layers.Conv2D(64, 7, 2, 'same', kernel_regularizer=self.regularizer)(inputs)
        x = K.layers.BatchNormalization()(x)
        x_0 = K.layers.ReLU(negative_slope=self.negative_slope)(x)
        x = K.layers.MaxPool2D(3, 2, padding='same')(x_0)

        x_1 = self.group_block(self.block, x, 64, 1, self.n_list[0])
        x_2 = self.group_block(self.block, x_1, 128, 2, self.n_list[1])
        x_3 = self.group_block(self.block, x_2, 256, 2, self.n_list[2])
        x_4 = self.group_block(self.block, x_3, 512, 2, self.n_list[3])

        return K.models.Model(inputs, x_4), [x_0, x_1, x_2, x_3, x_4]

    # @staticmethod
    def group_block(self, block, inputs, filters, strides, n):
        x = block(inputs, filters, strides, match_dims=True)
        for i in range(1, n):
            x = block(x, filters, 1)
        return x

    def residual_block(self, inputs, filters, strides, match_dims=False):
        shortcut = inputs
        if match_dims or strides!=1:
            shortcut = K.layers.Conv2D(filters, 1, strides, 'same', kernel_regularizer=self.regularizer)(inputs)
            shortcut = K.layers.BatchNormalization()(shortcut)

        x = K.layers.Conv2D(filters, 3, strides, 'same', kernel_regularizer=self.regularizer)(inputs)
        x = K.layers.BatchNormalization()(x)
        x = K.layers.ReLU(negative_slope=self.negative_slope)(x)
        x = K.layers.Conv2D(filters, 3, 1, 'same', kernel_regularizer=self.regularizer)(x)
        x = K.layers.BatchNormalization()(x)
        x = shortcut + x
        return K.layers.ReLU(negative_slope=self.negative_slope)(x)

    def bottleneck_block(self, inputs, filters, strides, match_dims=False):
        shortcut = inputs
        if match_dims or strides!=1:
            shortcut = K.layers.Conv2D(filters * 4, 1, strides, 'same', kernel_regularizer=self.regularizer)(inputs)
            shortcut = K.layers.BatchNormalization()(shortcut)

        x = K.layers.Conv2D(filters, 1, strides, 'same', kernel_regularizer=self.regularizer)(inputs)
        x = K.layers.BatchNormalization()(x)
        x = K.layers.ReLU(negative_slope=self.negative_slope)(x)
        x = K.layers.Conv2D(filters, 3, 1, 'same', kernel_regularizer=self.regularizer)(x)
        x = K.layers.BatchNormalization()(x)
        x = K.layers.ReLU(negative_slope=self.negative_slope)(x)
        x = K.layers.Conv2D(filters * 4, 1, 1, 'same', kernel_regularizer=self.regularizer)(x)
        x = K.layers.BatchNormalization()(x)
        x = shortcut + x
        return K.layers.ReLU(negative_slope=self.negative_slope)(x)
