import tensorflow as tf
import input

INPUT_FILENAME =  "/mnt/red/train/humanlike/preprocessed/train-otb-hq-2600-first51100-fixed"
OUTPUT_FILENAME = "/mnt/red/train/humanlike/preprocessed/train-otb-hq-2600-first51100-fixed-nokasparov"
record_iterator = tf.python_io.tf_record_iterator(path=INPUT_FILENAME)


SEARCHED_PLAYER = input.hash_32("Kasparov, Garry")

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

with tf.python_io.TFRecordWriter(OUTPUT_FILENAME) as writer:
    for string_record in record_iterator:        
        example = tf.train.Example()
        example.ParseFromString(string_record)
        if filter(example):
            writer.write(example.SerializeToString())
