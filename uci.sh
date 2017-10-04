#!/bin/bash
source activate tf13
cd /home/krisztian/projects/humanlike/turk && python -u uci.py --logdir=/mnt/red/train/humanlike/old/1/logdir/ 2>&1 | tee -a log.txt
