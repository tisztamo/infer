from __future__ import print_function
import numpy as np
import tensorflow as tf
import tensorflow.contrib
import random
import input
import model

FLAGS = tf.app.flags.FLAGS
NUM_BATCHES =2001
TOP_MAX = 5

label_strings = input.load_labels()

validation_filenames = input.find_files(FLAGS.data_dir, "*val*")
print("Found", len(validation_filenames), "validation files.")
random.shuffle(validation_filenames)

def accuracy(predictions, labels):
    retval = np.array([0.0] * TOP_MAX)
    for k in range(1, TOP_MAX + 1):
        top_k_preds = np.take(np.argpartition(predictions, -k), range(-k, 0), 1)
        valid_labels = np.argmax(labels, 1)
        good_pred_count = 0
        for i in range(top_k_preds.shape[0]):
            found = np.nonzero(top_k_preds[i] == valid_labels[i])
            if len(found[0]) > 0:
                good_pred_count += 1
        retval[k - 1] = (100.0 * good_pred_count
            / predictions.shape[0])
    return retval

#with tf.device('/cpu:0'):
validationfilenames = tf.placeholder(tf.string, shape=[None])
validationset = input.inputs(validation_filenames, shuffle=False)

iterator = tf.contrib.data.Iterator.from_structure(validationset.output_types,
                                validationset.output_shapes)

validation_init_op = iterator.make_initializer(validationset)

examples, labels, cp_scores = iterator.get_next()

logits, _ = model.model(examples)
prediction = tf.nn.softmax(logits)

labels = tf.one_hot(labels, model.NUM_LABELS, dtype=tf.float32)

saver = tf.train.Saver()

with tf.Session() as sess:
    checkpoint = tf.train.get_checkpoint_state(FLAGS.logdir)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print ("Successfully loaded:", checkpoint.model_checkpoint_path)
        sum_pred_acc = [0] * TOP_MAX
        sess.run(validation_init_op, feed_dict={validationfilenames: validation_filenames})
        for i in range(NUM_BATCHES):
            v_prediction, v_labels, v_cp_scores = sess.run([prediction, labels, cp_scores])
            pred_acc = accuracy(v_prediction, v_labels)
            sum_pred_acc = np.add(sum_pred_acc, pred_acc)
            mean_pred_acc = np.divide(sum_pred_acc, float(i + 1))
            print("Top-k (k=1.." + str(TOP_MAX) + ") accuracy after batch #" + str(i) + ":", mean_pred_acc, end="\r")
        print()
    else:
        print("No checkpoint found in logdir", FLAGS.logdir)
