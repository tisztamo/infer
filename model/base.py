import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from model import cnn_model, residual_model

FEATURE_PLANES = 12

model_impl = None

def get_model_impl():
    global model_impl
    if model_impl is None:
        model_impl = cnn_model.CNNModel()
        #model_impl = residual_model.ResidualModel()
    return model_impl

def feature_extractor(data):
    return get_model_impl().feature_extractor(data)

def policy_model(data, feature_tensor = None):
    return get_model_impl().policy_model(data, feature_tensor)

def result_model(data, feature_tensor = None):
    return get_model_impl().result_model(data, feature_tensor)
  
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
        #b = bias_variable([layer_size])
        linear = tf.matmul(h_input, W)# + b
        #pre_activation = slim.batch_norm(linear, activation_fn=None)
        pre_activation = linear
        if use_tanh_at_end and idx == len(hidden_layer_sizes) - 1:
            prev_output = tf.nn.tanh(pre_activation)
        else:
            prev_output = tf.nn.relu(pre_activation)

    return prev_output