import importlib
from collections import namedtuple

import tensorflow as tf
from architecture import resnet


def backbone_generator(params: namedtuple)-> tf.keras.models:
    if params.network.backbone.architecture == 'resnet':
        backbone_net = resnet.ResNet(params)

        return backbone_net


# def network_generator(params: namedtuple)-> tf.keras.models:
#     net_name = params.name
#     network_module = importlib.import_module('networks.'+net_name+'_network')
#     network = network_module.Net(params)
#     return network