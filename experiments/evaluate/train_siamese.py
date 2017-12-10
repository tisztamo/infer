import random
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib
import input
import model
from experiments.evaluate import siamese_model 

FLAGS = tf.app.flags.FLAGS

START_LEARNING_RATE = 0.01

def create_iterator(data_dir, mask="*"):
    train_filenames = input.find_files(data_dir, "train*")
    print("Found", len(train_filenames), "train files.")
    random.shuffle(train_filenames)

    filenames = tf.placeholder(tf.string, shape=[None])
    dataset = input.inputs(filenames)

    iterator = tf.contrib.data.Iterator.from_structure(dataset.output_types,
                                    dataset.output_shapes)

    init_op = iterator.make_initializer(dataset)

    return iterator, init_op, filenames, train_filenames

ww_iterator, ww_init, ww_filenames_op, ww_filenames = create_iterator(FLAGS.data_dir + "/whitewins", "train*")
bw_iterator, bw_init, bw_filenames_op, bw_filenames = create_iterator(FLAGS.data_dir + "/blackwins", "train*")

ww_examples, ww_labels, ww_results, ww_scores = ww_iterator.get_next()
bw_examples, bw_labels, bw_results, bw_scores = bw_iterator.get_next()


# Model
raw_input = tf.placeholder(tf.float32, shape=ww_examples[0].shape)
features = model.feature_extractor([raw_input])

score_diff = ww_scores - bw_scores

move_vars = tf.trainable_variables()
move_saver = tf.train.Saver(save_relative_paths=True, var_list=move_vars)

with tf.variable_scope("siamese"):
    double_feature_shape = [features.shape[0], features.shape[1] + features.shape[1]]
    siamese_input = tf.placeholder(tf.float32, shape=double_feature_shape)
    predicted_score_diff, trainables = siamese_model.score_diff_predictor(siamese_input)
    predicted_score_diff = 1500.0 * tf.squeeze(predicted_score_diff)

siamese_saver = tf.train.Saver(save_relative_paths=True, var_list=trainables)

# Losses
gt_score_diff = tf.placeholder(tf.float32, score_diff.shape)
#score_loss = tf.abs(tf.tanh((gt_score_diff - predicted_score_diff) / 100.0 ))#tf.losses.absolute_difference(gt_score_diff, predicted_score_diff)
#tf.losses.add_loss(score_loss)
score_loss = tf.losses.absolute_difference(predicted_score_diff, gt_score_diff)
loss = tf.losses.get_total_loss()


#Training
global_step = tf.Variable(0, name='global_step', trainable=False)
learning_rate = tf.train.exponential_decay(START_LEARNING_RATE, global_step,
                                           800, 0.99, staircase=True)
training_op = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step, var_list=trainables)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True

with tf.Session(config=config) as sess:
    merged_summaries = tf.summary.merge_all()
    #train_writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)
    sess.run(tf.global_variables_initializer())

    checkpoint = tf.train.get_checkpoint_state(FLAGS.logdir)
    if checkpoint and checkpoint.model_checkpoint_path:
        move_saver.restore(sess, checkpoint.model_checkpoint_path)
        print ("Successfully loaded feature extractor:", checkpoint.model_checkpoint_path)

    checkpoint = tf.train.get_checkpoint_state(FLAGS.logdir + "/siamese/")
    if checkpoint and checkpoint.model_checkpoint_path:
        siamese_saver.restore(sess, checkpoint.model_checkpoint_path)
        print ("Successfully loaded siamese net:", checkpoint.model_checkpoint_path)

    sess.run([ww_init, bw_init], feed_dict={ww_filenames_op: ww_filenames, bw_filenames_op: bw_filenames})

    l = 0
    BATCH_PER_PRINT=50
    avg_loss = None
    while True:
        ts = time.time()
        for i in range(BATCH_PER_PRINT):
            v_ww_examples, v_bw_examples, v_score_diff = sess.run([ww_examples, bw_examples, score_diff])

            flip = random.random() > 0.5
            if flip:
                example1 = v_bw_examples
                example2 = v_ww_examples
                v_score_diff = -v_score_diff
            else:
                example1 = v_ww_examples
                example2 = v_bw_examples
            v_features1 = sess.run(features, feed_dict={raw_input: example1[0]})
            v_features2 = sess.run(features, feed_dict={raw_input: example2[0]})

            v_siamese_input = np.concatenate([v_features1, v_features2], 1)
            _, l, step, lrate, v_p_score_diff, v_score_loss = sess.run([training_op, loss, global_step, learning_rate, predicted_score_diff, score_loss], feed_dict={siamese_input:v_siamese_input, gt_score_diff:v_score_diff})
            avg_loss = np.mean(l) if avg_loss is None else 0.98 * avg_loss + 0.02 * np.mean(l)

        elapsed = time.time() -ts
    
        #print("Loss at batch %d: %.2f, speed: %.1f examples/s, lr: %.7f" % (step, l, BATCH_PER_PRINT * input.BATCH_SIZE / elapsed, lrate))
        print("loss:", avg_loss, "step:", step, "speed: ", int(BATCH_PER_PRINT * input.BATCH_SIZE / elapsed), "lrate", lrate)
        if True:
            board = input.decode_board(v_ww_examples[0][0])
            print("First:")
            print(board)
            board = input.decode_board(v_bw_examples[0][0])
            print("Second:")
            print(board)
            print("Score diff:", v_score_diff[0], "predicted:", v_p_score_diff[0])
        #train_writer.add_summary(summaries, step)

        if step % 1000 == 0 and step > 0:
            #move_saver.save(sess, FLAGS.logdir + '/move', global_step=global_step)
            siamese_saver.save(sess, FLAGS.logdir + '/siamese/siamese', global_step=global_step)
            print("Model saved.")

