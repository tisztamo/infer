#!/bin/bash
cutechess-cli -fcp conf=stockfish tc=40/20 depth=8  -scp conf=humanlike st=12 -games 10 -concurrency 1 -pgnout stockfish_humanlike.pgn
