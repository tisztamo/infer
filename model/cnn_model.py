import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import model
from model import base

class CNNModel:
    def __init__(self):
        pass

    def feature_extractor(self, data):
        #input = slim.batch_norm(data[0], activation_fn=None)
        board = data[0]
        movelayers = data[1]

        input = tf.concat([board, movelayers], axis=3)

        h_conv1 = extractor_layer(input, 16, 32, 16, 16)
        input2 = tf.concat([input, h_conv1], axis=3)
        h_conv2 = extractor_layer(input2, 32, 384, 64, 32)
        input3 = tf.concat([input, h_conv2], axis=3)
        h_conv3 = extractor_layer(input3, 32, 512, 64, 32)
        input4 = tf.concat([input, h_conv1, h_conv3], axis=3)
        h_conv4 = extractor_layer(input4, 32, 768, 0, 0)
        input4b = tf.concat([input, h_conv1, h_conv4], axis=3)
        h_conv4b = extractor_layer(input4b, 32, 896, 0, 0)
        input5 = tf.concat([input, h_conv1, h_conv4b], axis=3)
        h_conv5 = extractor_layer(input5, 32, 1024, 0, 0)
        input6 = tf.concat([input,h_conv1, h_conv5], axis=3)
        h_conv6 = extractor_layer(input6, 32, 1024 + 256, 0, 0)

        h_flat = tf.reshape(h_conv6, [-1, (32 + 1024 + 256) * 64])

        return h_flat

    def policy_model(self, data, feature_tensor, layers_out):
        return base.model_head(data, feature_tensor, [2048, 2048, model.NUM_LABELS], layers_out)

    def result_model(self, data, feature_tensor, layers_out):
        return base.model_head(data, feature_tensor, [2048, 2048, 3], layers_out)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

def conv_layer(input, width, height, filter_num):
    num_input_filters = int(input.shape[3])
    w = base.weight_variable([width, height, num_input_filters, filter_num])
    conv = conv2d(input, w, 1)
    #normed = slim.batch_norm(conv, activation_fn=None)
    h = tf.nn.relu(conv)
    return h

def extractor_layer(input, num_1x1=0, num_3x3=0, num_5x5=0, num_7x7=0):
    num_input_filters = int(input.shape[3])
    print("Creating extractor layer for ", num_input_filters, "inputs.")
    extractors = []

    if num_1x1 > 0:
        h_1x1 = conv_layer(input, 1, 1, num_1x1)
        extractors.append(h_1x1)

    h_3x3 = None
    if num_3x3 > 0:
        h_3x3 = conv_layer(input, 3, 3, num_3x3)
        extractors.append(h_3x3)

    h_5x5 = None
    if num_5x5 > 0:
        h_5x5 = conv_layer(input, 5, 5, num_5x5)
        extractors.append(h_5x5)

    h_7x7 = None
    if num_7x7 > 0:
        h_7x7 = conv_layer(input, 7, 7, num_7x7)
        extractors.append(h_7x7)

    return tf.concat(extractors, 3)
    
def squeeze_layer(input, filter_num):
    num_input_filters = int(input.shape[3])
    print("Creating squeeze layer for ", num_input_filters, "inputs.")
    return conv_layer(input, 1, 1, filter_num)