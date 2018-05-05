#!/bin/bash
cutechess-cli -fcp conf=humanlike tc=40/400 -scp conf=stockfish depth=9 tc=1/1 -games 10 -concurrency 1 -pgnout stockfish_humanlike.pgn
