import numpy as np
import tensorflow as tf
import tensorflow.contrib
import random
import input
import model

FLAGS = tf.app.flags.FLAGS
NUM_BATCHES =100

validation_filenames = input.find_files(FLAGS.data_dir, "train*-of-*")
print("Found", len(validation_filenames), "validation files.")
random.shuffle(validation_filenames)

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

with tf.device('/cpu:0'):
    validationfilenames = tf.placeholder(tf.string, shape=[None])
    validationset = input.inputs(validation_filenames)

    iterator = tf.contrib.data.Iterator.from_structure(validationset.output_types,
                                    validationset.output_shapes)

    validation_init_op = iterator.make_initializer(validationset)

    filenames = tf.placeholder(tf.string, shape=[None])

    examples, labels = iterator.get_next()

    logits = model.model(examples)
    prediction = tf.nn.softmax(logits)

    labels = tf.one_hot(labels, model.NUM_LABELS, dtype=tf.float32)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        checkpoint = tf.train.get_checkpoint_state(FLAGS.logdir)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print ("Successfully loaded:", checkpoint.model_checkpoint_path)
            mean_acc = 0
            for i in range(NUM_BATCHES):
                sess.run(validation_init_op, feed_dict={validationfilenames: validation_filenames})
                v_prediction, v_labels = sess.run([prediction, labels])
                acc = accuracy(v_prediction, v_labels)
                mean_acc += acc
                print('Accuracy after batch #%d: %.1f%%' % (i, mean_acc / (i+1)))
            mean_acc = mean_acc / NUM_BATCHES
            print("-----")
            print("Model accuracy: %.1f%%" % mean_acc)
        else:
            print("No checkpoint found in logdir", FLAGS.logdir)
