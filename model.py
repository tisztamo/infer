import numpy as np
import tensorflow as tf

IMAGE_SIZE = 8
FEATURE_PLANES = 6
NUM_LABELS = 1972
HIDDEN = 2048

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.001, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")


def model(data):
    W_conv1 = weight_variable([5, 5, FEATURE_PLANES, 512])
    b_conv1 = bias_variable([512])
    h_conv1 = tf.nn.relu(conv2d(data[0], W_conv1, 1) + b_conv1)

    W_conv2 = weight_variable([5, 5, 512, 768])
    b_conv2 = bias_variable([768])
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 1) + b_conv2)

    W_conv3 = weight_variable([3, 3, 768, 1024])
    b_conv3 = bias_variable([1024])
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)

    h_flat = tf.reshape(h_conv3, [-1, 1024 * 64])

    cc = tf.cast([data[1]], tf.float32)
    cc = tf.transpose(cc)
    h_extra = tf.concat([h_flat, cc], axis=1)

    W_fc1 = weight_variable([1024 * 64 + 1, HIDDEN])
    b_fc1 = bias_variable([HIDDEN])
    h_fc1 = tf.nn.relu(tf.matmul(h_extra, W_fc1) + b_fc1)

    W_fc2 = weight_variable([HIDDEN, HIDDEN])
    b_fc2 = bias_variable([HIDDEN])
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    W_fc3 = weight_variable([HIDDEN, NUM_LABELS])
    b_fc3 = bias_variable([NUM_LABELS])

    readout = tf.matmul(h_fc2, W_fc3) + b_fc3
    return readout

