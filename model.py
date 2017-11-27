import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

IMAGE_SIZE = 8
FEATURE_PLANES = 12
NUM_HIDDEN_PLANES = 850
NUM_RES_BLOCKS = 7
NUM_LABELS = 1972

def summary(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(-0.001, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")


def conv_layer(input, width, height, filter_num, trainables=[]):
    num_input_filters = int(input.shape[3])
    w = weight_variable([width, height, num_input_filters, filter_num])
    trainables.append(w)
    b = bias_variable([filter_num])
    trainables.append(b)
    h = tf.nn.relu(conv2d(input, w, 1) + b)
    return h


def extractor_layer(input, num_1x1=0, num_3x3=0, num_5x5=0, num_7x7=0, trainables=[]):
    num_input_filters = int(input.shape[3])
    print("Creating extractor layer for ", num_input_filters, "inputs.")
    extractors = []

    if num_1x1 > 0:
        h_1x1 = conv_layer(input, 1, 1, num_1x1, trainables)
        extractors.append(h_1x1)

    h_3x3 = None
    if num_3x3 > 0:
        h_3x3 = conv_layer(input, 3, 3, num_3x3, trainables)
        extractors.append(h_3x3)

    h_5x5 = None
    if num_5x5 > 0:
        h_5x5 = conv_layer(input, 5, 5, num_5x5, trainables)
        extractors.append(h_5x5)

    h_7x7 = None
    if num_7x7 > 0:
        h_7x7 = conv_layer(input, 7, 7, num_7x7, trainables)
        extractors.append(h_7x7)

    return tf.concat(extractors, 3)
    
def squeeze_layer(input, filter_num, trainables=[]):
    num_input_filters = int(input.shape[3])
    print("Creating squeeze layer for ", num_input_filters, "inputs.")
    return conv_layer(input, 1, 1, filter_num, trainables)


def res_unit(input_layer, board, i):
    with tf.variable_scope("res_unit" + str(i)):
        inp = tf.concat([input_layer, board], axis=3)
        part = slim.conv2d(inp, NUM_HIDDEN_PLANES, [3,3], activation_fn=None)
        part = slim.batch_norm(part, activation_fn=None)
        part = tf.nn.relu(part)
        part = slim.conv2d(part, NUM_HIDDEN_PLANES, [3, 3], activation_fn=None)
        part = slim.batch_norm(part, activation_fn=None)
        part = input_layer + part
        output = tf.nn.relu(part)
        return output
    
def feature_extractor(data):
    layer1 = slim.conv2d(data[0], NUM_HIDDEN_PLANES, [3, 3], normalizer_fn=slim.batch_norm, scope='conv_' + str(0))
    layer1 = slim.batch_norm(layer1, activation_fn=None)
    layer1 = tf.nn.relu(layer1)
    for i in range(NUM_RES_BLOCKS):
        layer1 = res_unit(layer1, data[0], i)
    
    h_flat = tf.reshape(layer1, [-1, (NUM_HIDDEN_PLANES) * 64])
    return h_flat

def model_head(data, feature_tensor = None, hidden_layer_sizes = [512], use_tanh_at_end=False):
    """ data[0]: board representation

        feature_tensor: Extracted features as returned by feature_extractor.
    """

    if feature_tensor is None:
        feature_tensor = feature_extractor(data)

    cc = tf.reshape(data[0], [-1, 64 * FEATURE_PLANES])

    prev_output = feature_tensor
    for idx, layer_size in enumerate(hidden_layer_sizes):
        h_input = tf.concat([prev_output, cc], axis=1)
        W = weight_variable([int(h_input.shape[1]), layer_size])
        b = bias_variable([layer_size])
        pre_activation = tf.matmul(h_input, W) + b
        if use_tanh_at_end and idx == len(hidden_layer_sizes) - 1:
            prev_output = tf.nn.tanh(pre_activation)
        else:
            prev_output = tf.nn.relu(pre_activation)

    return prev_output

def policy_model(data, feature_tensor = None):
    return model_head(data, feature_tensor, [4096, 3072, 2048, NUM_LABELS])

def result_model(data, feature_tensor = None):
    return model_head(data, feature_tensor, [4096, 2048, 1024, 512, 1], use_tanh_at_end = True)
    