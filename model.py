import numpy as np
import tensorflow as tf

IMAGE_SIZE = 8
FEATURE_PLANES = 6
NUM_LABELS = 1972
HIDDEN = 2048

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

def feature_extractor(data):
    trainables = []
    W_conv1 = weight_variable([5, 5, FEATURE_PLANES, 512])
    trainables.append(W_conv1)
    b_conv1 = bias_variable([512])
    trainables.append(b_conv1)
    h_conv1 = tf.nn.relu(conv2d(data[0], W_conv1, 1) + b_conv1)

    # W_conv1p = weight_variable([1, 1, 512, 256])
    # trainables.append(W_conv1p)
    # b_conv1p = bias_variable([256])
    # trainables.append(b_conv1p)
    # h_conv1p = tf.nn.relu(conv2d(h_conv1, W_conv1p, 1) + b_conv1p)

    W_conv2 = weight_variable([5, 5, 512, 768])
    trainables.append(W_conv2)
    b_conv2 = bias_variable([768])
    trainables.append(b_conv2)
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 1) + b_conv2)

    # W_conv2p = weight_variable([1, 1, 512, 256])
    # trainables.append(W_conv2p)
    # b_conv2p = bias_variable([256])
    # trainables.append(b_conv2p)
    # h_conv2p = tf.nn.relu(conv2d(h_conv2, W_conv2p, 1) + b_conv2p)

    W_conv3 = weight_variable([3, 3, 768, 1024])
    trainables.append(W_conv3)
    b_conv3 = bias_variable([1024])
    trainables.append(b_conv3)
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)

    # W_conv3p = weight_variable([1, 1, 512, 256])
    # trainables.append(W_conv3p)
    # b_conv3p = bias_variable([256])
    # trainables.append(b_conv3p)
    # h_conv3p = tf.nn.relu(conv2d(h_conv3, W_conv3p, 1) + b_conv3p)

    h_flat = tf.reshape(h_conv3, [-1, 1024 * 64])

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

    cc = tf.cast([data[1]], tf.float32)
    cc = tf.transpose(cc)

    h_extra = tf.concat([feature_tensor, cc], axis=1)

    W_fc1 = weight_variable([1024 * 64 + 1, HIDDEN])
    trainables.append(W_fc1)
    b_fc1 = bias_variable([HIDDEN])
    trainables.append(b_fc1)
    h_fc1 = tf.nn.relu(tf.matmul(h_extra, W_fc1) + b_fc1)

    if dropout >= 0.0:
        h_dropout1 = tf.nn.dropout(h_fc1, 1.0 - dropout)
    else:
        h_dropout1  = h_fc1

    W_fc2 = weight_variable([HIDDEN, HIDDEN])
    trainables.append(W_fc2)
    b_fc2 = bias_variable([HIDDEN])
    trainables.append(b_fc2)
    h_fc2 = tf.nn.relu(tf.matmul(h_dropout1, W_fc2) + b_fc2)

    if dropout >= 0.0:
        h_dropout2 = tf.nn.dropout(h_fc2, 1.0 - dropout)
    else:
        h_dropout2  = h_fc2

    W_fc3 = weight_variable([HIDDEN, NUM_LABELS])
    trainables.append(W_fc3)
    b_fc3 = bias_variable([NUM_LABELS])
    trainables.append(b_fc3)

    readout = tf.matmul(h_dropout2, W_fc3) + b_fc3
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
