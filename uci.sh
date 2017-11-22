#!/bin/bash
python -u uci.py --logdir=/mnt/red/train/humanlike/logdir/ --gpu=false --play_first_intuition=true --use_back_engine=true $@ 2>&1 | tee -a log.txt
