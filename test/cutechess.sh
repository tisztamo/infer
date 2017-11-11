#!/bin/bash
cutechess-cli -fcp conf=humanlike tc=40/140  -scp conf=humanlike tc=40/140 -both depth=13 -games 10 -concurrency 1 -pgnout humanlike_humanlike.pgn
