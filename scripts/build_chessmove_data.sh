#!/bin/bash
python build_chessmove_data.py --train_dir=/mnt/red/train/humanlike/source/train/ --output_dir=/mnt/red/train/humanlike/preprocessed/whitewins/ --validation_dir=/mnt/red/train/humanlike/source/validation/ "--filter_player=-Kasparov, Garry" --omit_draws=true --omit_blackwins=true --gpu=false --disable_cp=false --skip_plies=20 --filter_black=true

