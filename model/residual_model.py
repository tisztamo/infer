import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import model
from model import base

NUM_RES_BLOCKS =12 
NUM_HIDDEN_PLANES = 850
class ResidualModel:
    def __init__(self):
        pass

    def feature_extractor(self, data):
        layer1 = slim.conv2d(data[0], NUM_HIDDEN_PLANES, [3, 3], scope='conv_' + str(0))
        #layer1 = slim.batch_norm(layer1, activation_fn=None)
        layer1 = tf.nn.relu(layer1)
        for i in range(NUM_RES_BLOCKS):
            layer1 = res_unit(layer1, data[0], i)
        
        h_flat = tf.reshape(layer1, [-1, (NUM_HIDDEN_PLANES) * 64])
        return h_flat

    def policy_model(self, data, feature_tensor, layers_out = None):
        return base.model_head(data, feature_tensor, [3072, 2048, model.NUM_LABELS], layers_out)

    def result_model(self, data, feature_tensor, layers_out = None):
        return base.model_head(data, feature_tensor, [4096, 2048, 1024, 512, 3], layers_out)

def res_unit(input_layer, board, i):
    with tf.variable_scope("res_unit" + str(i)):
        inp = tf.concat([input_layer, board], axis=3)
        part = slim.conv2d(inp, NUM_HIDDEN_PLANES, [3,3], activation_fn=None)
        #part = slim.batch_norm(part, activation_fn=None)
        part = tf.nn.relu(part)
        part = slim.conv2d(part, NUM_HIDDEN_PLANES, [3, 3], activation_fn=None)
        #part = slim.batch_norm(part, activation_fn=None)
        part = input_layer + part
        output = tf.nn.relu(part)
        return output

