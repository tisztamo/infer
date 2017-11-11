import numpy as np
import tensorflow as tf

IMAGE_SIZE = 8
FEATURE_PLANES = 6
NUM_LABELS = 1972
HIDDEN = 3072

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
    
def feature_extractor(data):
    trainables = []

    h_conv1 = extractor_layer(data[0], 32, 256, 256, 256, trainables)
    h_conv2 = extractor_layer(h_conv1, 32, 256, 256, 256, trainables)
    h_conv3 = extractor_layer(h_conv2, 32, 512, 256, 0, trainables)
    h_conv4 = extractor_layer(h_conv3, 32, 768, 512, 256, trainables)

    h_flat = tf.reshape(h_conv4, [-1, 1568 * 64])

    return h_flat, trainables

def model(data, feature_tensor=None, trainables = [], dropout = 0.0):
    """ data[0]: board representation
        data[1]: turn
        data[2]: player

        feature_tensor: Extracted features as returned by feature_extractor.
        dropout: The probability of an activation to be _dropped_ on the dropout layer.
    """
    if feature_tensor is None:
        feature_tensor, trainables = feature_extractor(data)

    #cc = tf.cast([data[1]], tf.float32)
    #cc = tf.transpose(cc)

    h_extra = feature_tensor #tf.concat([feature_tensor, cc], axis=1)

    W_fc1 = weight_variable([1568 * 64, HIDDEN])
    trainables.append(W_fc1)
    b_fc1 = bias_variable([HIDDEN])
    trainables.append(b_fc1)
    h_fc1 = tf.nn.relu(tf.matmul(h_extra, W_fc1) + b_fc1)

    if dropout >= 0.0:
        h_dropout1 = tf.nn.dropout(h_fc1, 1.0 - dropout)
    else:
        h_dropout1  = h_fc1


    #Not working, fails to learn
    # W_bottleneck = weight_variable([HIDDEN, 500])
    # trainables.append(W_bottleneck)
    # b_bottleneck = bias_variable([500])
    # trainables.append(b_bottleneck)
    # h_bottleneck = tf.nn.relu(tf.matmul(h_dropout1, W_bottleneck) + b_bottleneck)
    # if dropout >= 0.0:
    #     h_dropout_bottleneck = tf.nn.dropout(h_bottleneck, 1.0 - dropout)
    # else:
    #     h_dropout_bottleneck  = h_bottleneck


    W_fc2 = weight_variable([HIDDEN, HIDDEN])
    trainables.append(W_fc2)
    b_fc2 = bias_variable([HIDDEN])
    trainables.append(b_fc2)
    h_fc2 = tf.nn.relu(tf.matmul(h_dropout1, W_fc2) + b_fc2)

    if dropout >= 0.0:
        h_dropout2 = tf.nn.dropout(h_fc2, 1.0 - dropout)
    else:
        h_dropout2  = h_fc2

    W_fc3 = weight_variable([HIDDEN, HIDDEN])
    trainables.append(W_fc3)
    b_fc3 = bias_variable([HIDDEN])
    trainables.append(b_fc3)
    h_fc3 = tf.nn.relu(tf.matmul(h_dropout2, W_fc3) + b_fc3)

    if dropout >= 0.0:
        h_dropout3 = tf.nn.dropout(h_fc3, 1.0 - dropout)
    else:
        h_dropout3  = h_fc3

    W_fc4 = weight_variable([HIDDEN, NUM_LABELS])
    trainables.append(W_fc4)
    b_fc4 = bias_variable([NUM_LABELS])
    trainables.append(b_fc4)

    readout = tf.matmul(h_dropout3, W_fc4) + b_fc4
    return readout, trainables

def value_model(data, feature_tensor=None, trainables=[]):
    if feature_tensor is None:
        feature_tensor, trainables = feature_extractor(data)

    value_extradata = tf.cast([data[1]], tf.float32)
    value_extradata = tf.transpose(value_extradata)
    value_h_extra = tf.concat([feature_tensor, value_extradata], axis=1)

    value_W_fc1 = weight_variable([1024 * 64 + 1, HIDDEN])
    trainables.append(value_W_fc1)
    value_b_fc1 = bias_variable([HIDDEN])
    trainables.append(value_b_fc1)
    value_h_fc1 = tf.nn.tanh(tf.matmul(value_h_extra, value_W_fc1) + value_b_fc1)

    value_W_fc2 = weight_variable([HIDDEN, HIDDEN])
    trainables.append(value_W_fc2)
    value_b_fc2 = bias_variable([HIDDEN])
    trainables.append(value_b_fc2)
    value_h_fc2 = tf.nn.tanh(tf.matmul(value_h_fc1, value_W_fc2) + value_b_fc2)

    value_W_fc3 = weight_variable([HIDDEN, 1])
    trainables.append(value_W_fc3)
    value_b_fc3 = bias_variable([1])
    trainables.append(value_b_fc3)

    readout = tf.matmul(value_h_fc2, value_W_fc3) + value_b_fc3
    return tf.reshape(readout, [-1]), trainables
