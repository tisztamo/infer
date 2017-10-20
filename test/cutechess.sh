#!/bin/bash
cutechess-cli -fcp conf=stockfish tc=40/20  -scp conf=humanlike tc=40/140 -games 10 -concurrency 1 -pgnout stockfish_humanlike.pgn
