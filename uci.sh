#!/bin/bash
python -u uci.py --logdir=/mnt/red/train/humanlike/logdir/ 2>&1 | tee -a log.txt
