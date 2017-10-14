import random
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib
import input
import model

FLAGS = tf.app.flags.FLAGS

TRANSFER_LEARNING = True

train_filenames = input.find_files(FLAGS.data_dir, "*train*")
print("Found", len(train_filenames), "train files.")
random.shuffle(train_filenames)

filenames = tf.placeholder(tf.string, shape=[None])
dataset = input.inputs(filenames)

iterator = tf.contrib.data.Iterator.from_structure(dataset.output_types,
                                   dataset.output_shapes)

training_init_op = iterator.make_initializer(dataset)

examples, labels, cp_scores = iterator.get_next()

features, trainables = model.feature_extractor(examples)
logits, trainables = model.model(examples, features, trainables)
labels = tf.one_hot(labels, model.NUM_LABELS, dtype=tf.float32)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

#predicted_scores, score_trainables = model.value_model(examples, features, [])
#sq = tf.squared_difference(predicted_scores, cp_scores)
#score_loss = tf.reduce_mean(sq)
#model.summary(score_loss)

trained_vars = trainables[-2:] if TRANSFER_LEARNING else trainables
training_op = tf.train.AdagradOptimizer(0.001).minimize(loss, var_list=trained_vars)

#score_optimizer = tf.train.AdagradOptimizer(0.0001)#.minimize(score_loss)#, var_list=score_trainables
#score_gradients, score_variables = zip(*score_optimizer.compute_gradients(score_loss))
#score_gradients, _ = tf.clip_by_global_norm(score_gradients, 1.0)
#score_training_op = score_optimizer.apply_gradients(zip(score_gradients, score_variables))

saver = tf.train.Saver(var_list=trainables, save_relative_paths=True)
#score_saver = tf.train.Saver(var_list=score_trainables, save_relative_paths=True)
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

    #score_checkpoint = tf.train.get_checkpoint_state(FLAGS.logdir + "/score")
    #if score_checkpoint and score_checkpoint.model_checkpoint_path:
    #    score_saver.restore(sess, score_checkpoint.model_checkpoint_path)
    #    print ("Successfully loaded:", score_checkpoint.model_checkpoint_path)

    step = 0
    sess.run(training_init_op, feed_dict={filenames: train_filenames})

    l = 0
    score_l = 0
    while True:
        ts = time.time()
        for i in range(100):
            _, l = sess.run([training_op, loss])
            #summaries, sq_v, pred_cp, gt_cp, _, score_l = sess.run([merged_summaries, sq,predicted_scores, cp_scores, score_training_op, score_loss])
        elapsed = time.time() -ts
    
        step += 100
        print("Prediction / Score loss at batch %d: %.2f / %.3f, speed: %.1f examples/s" % (step, l, score_l, 100 * input.BATCH_SIZE / elapsed))
        #for i in range(10):
        #    print(pred_cp[i], gt_cp[i], abs(pred_cp[i] - gt_cp[i]), (pred_cp[i] + 100) / (gt_cp[i] + 100))
        #train_writer.add_summary(summaries, step)

        if step % 1000 == 0 and step > 0:
            saver.save(sess, FLAGS.logdir + '/move', global_step=step)
            #score_saver.save(sess, FLAGS.logdir + '/score/score', global_step=step)
            print("Model saved.")

