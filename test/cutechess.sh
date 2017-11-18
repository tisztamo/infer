#!/bin/bash
cutechess-cli -fcp conf=humanlike tc=inf -scp conf=stockfish tc=40/40 -both depth=10 -games 20 -concurrency 1 -pgnout stockfish_humanlike.pgn
