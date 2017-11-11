import tensorflow as tf
import numpy as np
import input

INPUT_FILENAME =  "/mnt/red/train/humanlike/preprocessed/train-otb-hq-2600-55200-end-nokasparov"
OUTPUT_FILENAME = "/mnt/red/train/humanlike/preprocessed/train-otb-hq-2600-55200-end-nokasparov-onecolor"
record_iterator = tf.python_io.tf_record_iterator(path=INPUT_FILENAME)


tf.app.flags.DEFINE_string('eval_depth', '5',
                           'Depth to eval position using the external engine')
FLAGS = tf.app.flags.FLAGS

SEARCHED_PLAYER = input.hash_32("Kasparov, Garry")


label_strings, _ = input.load_labels()

def filter_by_player(example):
    player_hash = example.features.feature["move/player"].int64_list.value[0]
    return player_hash == SEARCHED_PLAYER

def filter(example):
    # cp_score = int(example.features.feature['board/cp_score']
    #                              .int64_list.value[0])
    # uci = (example.features.feature['move/uci']
    #                             .bytes_list
    #                             .value[0])
    # best_uci = (example.features.feature['board/best_uci']
    #                             .bytes_list
    #                             .value[0])
    # complexity = (example.features.feature['board/complexity'].int64_list.value[0])
    return not filter_by_player(example)

def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float32_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes (string) features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def switch_if_black(example):
    if example.features.feature["move/turn"].int64_list.value[0] == 1:
        return example
    board = example.features.feature["board/sixlayer"].float_list.value
    board = np.reshape(board, (6, 8, 8))
    board = np.where(board == 0.0, 0.0, -board)
    board = np.flip(board, 2)

    label = example.features.feature['move/label'].int64_list.value[0]
    label_str = input.sideswitch_label(label_strings[label])
    label = label_strings.index(label_str)

    feature_desc = {
        'board/sixlayer': tf.train.Feature(float_list=tf.train.FloatList(value=np.ravel(board))),
        'board/fen': _bytes_feature(example.features.feature['board/fen'].bytes_list.value[0]),
        'move/player': _int64_feature(example.features.feature['move/player'].int64_list.value[0]),
        'move/turn': _int64_feature(1),
        'move/label': _int64_feature(label)
    }
    if FLAGS.disable_cp != "false" or int(FLAGS.eval_depth) > 0:
        key = "board/cp_score/" + FLAGS.eval_depth
        feature_desc[key] = _int64_feature(example.features.feature[key].int64_list.value[0])

    example = tf.train.Example(features=tf.train.Features(feature=feature_desc)) 
    return example

with tf.python_io.TFRecordWriter(OUTPUT_FILENAME) as writer:
    for string_record in record_iterator:        
        example = tf.train.Example()
        example.ParseFromString(string_record)
        if filter(example):
            example = switch_if_black(example)
            writer.write(example.SerializeToString())
