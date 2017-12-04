#!/bin/bash
cutechess-cli -fcp conf=humanlike tc=inf -scp conf=turk-release tc=40/40 -both depth=15 -games 4 -concurrency 1 -pgnout stockfish_humanlike.pgn
