import random
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib
import input
import model
import flags

FLAGS = flags.FLAGS

START_LEARNING_RATE = 0.007

# Inputs
train_filenames = input.find_files(FLAGS.data_dir, "train-*")
print("Found", len(train_filenames), "train files.")
random.shuffle(train_filenames)

filenames = tf.placeholder(tf.string, shape=[None])
dataset = input.inputs(filenames)

iterator = tf.data.Iterator.from_structure(dataset.output_types,
                                   dataset.output_shapes)

training_init_op = iterator.make_initializer(dataset)

examples, labels = iterator.get_next()

labels = tf.one_hot(labels, model.NUM_LABELS, dtype=tf.float32)

# Model
features = model.feature_extractor(examples)
logits = model.policy_model(examples, features)

# Losses
policy_loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels)
loss = tf.losses.get_total_loss()

#Training
global_step = tf.Variable(0, name='global_step', trainable=False)
learning_rate = tf.train.exponential_decay(START_LEARNING_RATE, global_step,
                                           1000, 0.995, staircase=True)
#Adagrad volt eredetileg!
training_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=global_step)

saver = tf.train.Saver(save_relative_paths=True)
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
    BATCH_PER_PRINT=10
    while True:
        ts = time.time()

        result_l = 0
        for i in range(BATCH_PER_PRINT):
            _, policy_l, l, step, lrate = sess.run([training_op, policy_loss, loss, global_step, learning_rate])
        elapsed = time.time() -ts
    
        print("Loss at batch %d: %.2f + %.2f = %.2f, speed: %.1f examples/s, lr: %.7f" % (step, policy_l, result_l, l, BATCH_PER_PRINT * input.BATCH_SIZE / elapsed, lrate))
        #train_writer.add_summary(summaries, step)

        if step % 1000 == 0 and step > 0:
            saver.save(sess, FLAGS.logdir + '/move', global_step=global_step)
            print("Model saved.")

