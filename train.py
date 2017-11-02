import random
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib
import input
import model

FLAGS = tf.app.flags.FLAGS

TRANSFER_LEARNING = False
START_LEARNING_RATE = 0.1 if not TRANSFER_LEARNING else 0.0002

train_filenames = input.find_files(FLAGS.data_dir, "train-otb-hq-*2400*")
print("Found", len(train_filenames), "train files.")
random.shuffle(train_filenames)

filenames = tf.placeholder(tf.string, shape=[None])
dataset = input.inputs(filenames)

iterator = tf.contrib.data.Iterator.from_structure(dataset.output_types,
                                   dataset.output_shapes)

training_init_op = iterator.make_initializer(dataset)

examples, labels, cp_scores = iterator.get_next()

features, trainables = model.feature_extractor(examples)
logits, trainables = model.model(examples, features, trainables, dropout=None)
labels = tf.one_hot(labels, model.NUM_LABELS, dtype=tf.float32)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

global_step = tf.Variable(0, name='global_step', trainable=False)
learning_rate = tf.train.exponential_decay(START_LEARNING_RATE, global_step,
                                           1000, 0.98, staircase=False)
trained_vars = trainables[-6:] if TRANSFER_LEARNING else trainables
training_op = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step, var_list=trained_vars)

saveables = list(trainables)
if not TRANSFER_LEARNING:
    saveables.append(global_step)

saver = tf.train.Saver(var_list=saveables, save_relative_paths=True)
config = tf.ConfigProto()
config.gpu_options.allow_growth=True

with tf.Session(config=config) as sess:
    merged_summaries = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)
    sess.run(tf.global_variables_initializer())

    checkpoint = tf.train.get_checkpoint_state(FLAGS.logdir)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print ("Successfully loaded:", checkpoint.model_checkpoint_path)

    sess.run(training_init_op, feed_dict={filenames: train_filenames})

    l = 0
    while True:
        ts = time.time()
        for i in range(100):
            _, l, step, lrate = sess.run([training_op, loss, global_step, learning_rate])
        elapsed = time.time() -ts
    
        print("Learning rate", lrate)
        print("Loss at batch %d: %.2f, speed: %.1f examples/s" % (step, l, 100 * input.BATCH_SIZE / elapsed))
        #train_writer.add_summary(summaries, step)

        if step % 1000 == 0 and step > 0:
            saver.save(sess, FLAGS.logdir + '/move', global_step=global_step)
            print("Model saved.")

