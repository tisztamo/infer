import tensorflow as tf

tf.app.flags.DEFINE_string('data_dir', '/mnt/red/inferdata/preprocessed/',
                           'Preprocessed training data directory')
tf.app.flags.DEFINE_string('labels_file', 'labels.txt',
                           'List of all labels (uci move notation)')
tf.app.flags.DEFINE_string('logdir', '/mnt/red/inferdata/logdir',
                           'Directory to store network parameters and training logs')
tf.app.flags.DEFINE_string('disable_cp', 'false',
                           'Do not load of cp_score field from the tfrecord data files')
tf.app.flags.DEFINE_string('repeat_dataset', 'false',
                           'Repeat input dataset indefinitely')
tf.app.flags.DEFINE_string('gpu', 'true',
                           'Use the GPU')

tf.app.flags.DEFINE_string('train_dir', '../data/train/',
                           'Training data directory')
tf.app.flags.DEFINE_string('validation_dir', '../data/validation/',
                           'Validation data directory')
tf.app.flags.DEFINE_string('eval_depth', '10',
                           'Depth to eval position using the external engine')
tf.app.flags.DEFINE_string('engine_exe', '../stockfish-8-linux/Linux/stockfish_8_x64',
                           'UCI engine executable')
tf.app.flags.DEFINE_string('skip_games', '0',
                           'Skip the first N games')
tf.app.flags.DEFINE_string('filter_player', '',
                           'Process only moves of the given player, or omit the player if the option starts with "-"')
tf.app.flags.DEFINE_string('prune_opening', 'true',
                           'Drop opening moves randomly to lower bias caused by repeating opening boards')                           
tf.app.flags.DEFINE_string('omit_draws', 'false',
                           'Omit games that ended in a draw')


tf.app.flags.DEFINE_string('play_first_intuition', 'false',
                           'Play the raw output of the policy network')
tf.app.flags.DEFINE_string('use_back_engine', 'true',
                           'Whether to use external engine for static (leaf) evaluation')
tf.app.flags.DEFINE_string('back_engine_exe', '../stockfish-8-linux/Linux/stockfish_8_x64_modern',
                           'External engine executable')
tf.app.flags.DEFINE_string('back_engine_depth', '8',
                           'External engine search depth')
tf.app.flags.DEFINE_string('search_depth', '1',
                           'Search depth')

FLAGS = tf.app.flags.FLAGS