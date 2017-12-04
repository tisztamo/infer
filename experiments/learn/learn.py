import random
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib
import input
import model
import experiments
import experiments.learn
from experiments.learn import memory_head

FLAGS = tf.app.flags.FLAGS

START_LEARNING_RATE = 0.005

# Inputs
train_filenames = input.find_files(FLAGS.data_dir, "train-otb*")
print("Found", len(train_filenames), "train files.")
random.shuffle(train_filenames)

filenames = tf.placeholder(tf.string, shape=[None])
dataset = input.inputs(filenames)

iterator = tf.contrib.data.Iterator.from_structure(dataset.output_types,
                                   dataset.output_shapes)

training_init_op = iterator.make_initializer(dataset)

examples, labels, results = iterator.get_next()

labels = tf.cast(labels, tf.int32)
#labels = tf.one_hot(labels, model.NUM_LABELS, dtype=tf.float32)

# Model
features = model.feature_extractor(examples)
layers = []
nn_logits = model.policy_model(examples, features, layers)


saver = tf.train.Saver(save_relative_paths=True)

memory = memory_head.MemoryHead(32768 * 8, model.NUM_LABELS)
memory_input = layers[1]#tf.one_hot(labels, model.NUM_LABELS, dtype=tf.float32)#layers[2]
logits, mask, teacher_loss = memory.policy_model(examples, memory_input, labels)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True

fullsaver = tf.train.Saver(save_relative_paths=True)

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    checkpoint = tf.train.get_checkpoint_state(FLAGS.logdir)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print ("Successfully loaded:", checkpoint.model_checkpoint_path)

    sess.run(training_init_op, feed_dict={filenames: train_filenames})

    l = 0
    BATCH_PER_PRINT=10
    step = 0
    while True:
        try:
            ts = time.time()
            for i in range(BATCH_PER_PRINT):
                v_labels, v_memory_input, v_logits, v_mask, v_teacher_loss = sess.run([labels, memory_input, logits, mask, teacher_loss])
            elapsed = time.time() -ts
            accuracy = np.mean(v_labels==v_logits)
            print("Loss at batch %d: %.4f, accuracy: %.2f speed: %.1f examples/s" % (step, v_teacher_loss, accuracy, BATCH_PER_PRINT * input.BATCH_SIZE / elapsed))
            step += 1
            if step % 100 == 0 and step > 0:
                fullsaver.save(sess, FLAGS.logdir + '/ memory', global_step=0)
                print("Model saved.")
        except Exception as e:
            raise e

    fullsaver.save(sess, FLAGS.logdir + '/memory', global_step=0)

