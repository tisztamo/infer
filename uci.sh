#!/bin/bash
python -u uci.py --logdir=/mnt/red/train/humanlike/logdir/residual --gpu=true --play_first_intuition=true --use_back_engine=false $@ 2>&1 | tee -a log.txt
