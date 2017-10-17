import tensorflow as tf
import input

INPUT_FILENAME =  "/mnt/red/train/humanlike/preprocessed/train-otb-hq-2600-51200-55200"
OUTPUT_FILENAME = "/mnt/red/train/humanlike/preprocessed/train-otb-hq-2600-51200-55200-fixed"
record_iterator = tf.python_io.tf_record_iterator(path=INPUT_FILENAME)

with tf.python_io.TFRecordWriter(OUTPUT_FILENAME) as writer:
    for string_record in record_iterator:        
        example = tf.train.Example()
        example.ParseFromString(string_record)
        
        # cp_score = int(example.features.feature['board/cp_score']
        #                              .int64_list.value[0])
        # uci = (example.features.feature['move/uci']
        #                             .bytes_list
        #                             .value[0])
        # best_uci = (example.features.feature['board/best_uci']
        #                             .bytes_list
        #                             .value[0])
        # complexity = (example.features.feature['board/complexity'].int64_list.value[0])
        # if uci == "e2e4":
        #     print(uci, best_uci,cp_score,complexity)
        writer.write(example.SerializeToString())
