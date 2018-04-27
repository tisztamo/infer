#!/bin/bash
cutechess-cli -fcp conf=humanlike -scp conf=stockfish -both tc=40/40 -games 10 -concurrency 1 -pgnout stockfish_humanlike.pgn
