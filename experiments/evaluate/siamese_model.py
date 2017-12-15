import tensorflow as tf
import model

def score_diff_predictor(shape):
    with tf.variable_scope("siamese"):
        layer_sizes = [1024, 768, 512, 384, 256, 192, 128, 64, 1]
        double_feature_shape = [shape[0], shape[1] + shape[1]]
        siamese_input = tf.placeholder(tf.float32, shape=double_feature_shape)

        layer_vars = []
        nn = model.base.model_head(None, siamese_input, layer_sizes, layer_vars, True)
        return siamese_input, nn, layer_vars