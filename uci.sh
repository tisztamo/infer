#!/bin/bash
python -u uci.py --logdir=/mnt/red/inferdata/logdir --gpu=false --play_first_intuition=false $@ 2>&1 | tee -a log.txt
