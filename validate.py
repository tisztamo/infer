from __future__ import print_function
import numpy as np
import tensorflow as tf
import tensorflow.contrib
import random
import input
import model

FLAGS = tf.app.flags.FLAGS
NUM_BATCHES = 2001
TOP_MAX = 5

label_strings, _ = input.load_labels()

validation_filenames = input.find_files(FLAGS.data_dir, "*trai*")
print("Found", len(validation_filenames), "validation files.")
random.shuffle(validation_filenames)

def accuracy(predictions, labels, fens=None):
    retval = np.array([0.0] * TOP_MAX)
    for k in range(1, TOP_MAX + 1):
        top_k_preds = np.take(np.argpartition(predictions, -k), range(-k, 0), 1)
        valid_labels = np.argmax(labels, 1)
        good_pred_count = 0
        for i in range(top_k_preds.shape[0]):
            found = np.nonzero(top_k_preds[i] == valid_labels[i])
            if len(found[0]) > 0:
                good_pred_count += 1
            if fens is not None:
                if k == 1:
                   csv_out.write(str(fens[i]) + "," + str(label_strings[valid_labels[i]]) + "," + str(label_strings[top_k_preds[i][0]]) + "\n")
        retval[k - 1] = (100.0 * good_pred_count
            / predictions.shape[0])
    return retval

def result_accuracy(result_predictions, results):
    threshold = 0.001
    num_correct_preds = np.array([0] * 3)
    num_preds = np.array([0] * 3)
    for i, pred_logits in enumerate(result_predictions):
        pred = np.argmax(pred_logits) - 1
        num_preds[pred + 1] += 1
        if results[i] == pred:
            num_correct_preds[pred + 1] += 1
    return num_correct_preds, num_preds

with tf.device(input.device):
    validationfilenames = tf.placeholder(tf.string, shape=[None])
    validationset = input.inputs(validation_filenames, shuffle=False)

    iterator = tf.contrib.data.Iterator.from_structure(validationset.output_types,
                                    validationset.output_shapes)

    validation_init_op = iterator.make_initializer(validationset)

    examples, labels, results = iterator.get_next()

    features = model.feature_extractor(examples)
    logits = model.policy_model(examples, features)
    result_prediction = model.result_model(examples, features)

    prediction = tf.nn.softmax(logits)    

    labels = tf.one_hot(labels, model.NUM_LABELS, dtype=tf.float32)

    saver = tf.train.Saver()

csv_out = open("validation.csv", "w")

try:
    with tf.Session() as sess:
        checkpoint = tf.train.get_checkpoint_state(FLAGS.logdir)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print ("Successfully loaded:", checkpoint.model_checkpoint_path)
            sum_pred_acc = [0.0] * TOP_MAX
            sum_res_correct_preds = np.array([0.0] * 3)
            sum_res_preds = np.array([1] * 3)
            sess.run(validation_init_op, feed_dict={validationfilenames: validation_filenames})
            for i in range(NUM_BATCHES):
                v_prediction, v_result_prediction, v_labels, v_results = sess.run([prediction, result_prediction, labels, results])
                pred_acc = accuracy(v_prediction, v_labels)
                res_correct_preds, res_preds = result_accuracy(v_result_prediction, v_results)
                sum_res_correct_preds += res_correct_preds
                sum_res_preds += res_preds
                sum_pred_acc = np.add(sum_pred_acc, pred_acc)
                mean_pred_acc = np.divide(sum_pred_acc, float(i + 1))
                mean_res_pred_acc = sum_res_correct_preds / sum_res_preds * 100.0
                print("Batch #" + str(i) + " Result [-1,0,1]: ", mean_res_pred_acc, "Move top-k (k=1.." + str(TOP_MAX) + "):", mean_pred_acc, end="\r")
            print()
        else:
            print("No checkpoint found in logdir", FLAGS.logdir)
    csv_out.close()
except Exception as e:
    csv_out.close()
    raise e

