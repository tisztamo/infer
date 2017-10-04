#!/bin/bash
source activate tf13
cd /home/krisztian/projects/humanlike/turk && python -u uci.py 2>&1 | tee -a log.txt
