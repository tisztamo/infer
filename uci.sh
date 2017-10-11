#!/bin/bash
cd /home/k.o/humanlike/turk && python -u uci.py --logdir=../logdir 2>&1 | tee -a log.txt
