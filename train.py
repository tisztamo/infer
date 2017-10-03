import numpy as np
import tensorflow as tf
import tensorflow.contrib
import random
import time
import os
import fnmatch
import input
import model

FLAGS = tf.app.flags.FLAGS

def find_files(directory, pattern):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files


train_filenames = find_files(FLAGS.data_dir, "*-of-*")
print("Found", len(train_filenames), "train files.")
random.shuffle(train_filenames)

filenames = tf.placeholder(tf.string, shape=[None])
dataset = input.inputs(filenames)

iterator = tf.contrib.data.Iterator.from_structure(dataset.output_types,
                                   dataset.output_shapes)

training_init_op = iterator.make_initializer(dataset)

examples, labels = iterator.get_next()

logits = model.model(examples)

labels = tf.one_hot(labels, model.NUM_LABELS, dtype=tf.float32)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

training_op = tf.train.AdagradOptimizer(0.001).minimize(loss)

saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth=True

with tf.Session(config=config) as sess:
    checkpoint = tf.train.get_checkpoint_state(FLAGS.logdir)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print ("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
    step = 0
    sess.run(training_init_op, feed_dict={filenames: train_filenames})

    while True:
        ts = time.time()
        for i in range(100):
            _, l = sess.run([training_op, loss])
        elapsed = time.time() -ts
    
        step += 100
        print("Loss at batch %d: %.2f, speed: %.1f examples/s" % (step, l, 100 * input.BATCH_SIZE / elapsed))

        if step % 5000 == 0 and step > 0:
            saver.save(sess, FLAGS.logdir + '/move', global_step=step)
            print("Model saved.")

