#!/bin/bash
source activate tf13
cd /home/krisztian/projects/humanlike/network && python -u uci.py 2>&1 | tee -a log.txt
