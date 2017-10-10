from __future__ import print_function

import sys
import fnmatch
from nbstreamreader import NonBlockingStreamReader
import engine
import numpy as np
import chess

board = chess.Board()
turk = engine.Engine()
remembered_move = None

def parse_position(line):
    global board
    words = line.split(" ")
    if words[1] == "startpos":
        board = chess.Board()
        for move in words[3:]:
            board.push_uci(move)
    elif words[1] == "fen":
        fen = " ".join(words[2:]).replace(" -1 ", " - ")
        print("fen", fen)
        board = chess.Board(fen)


def handle_go(line):
    global remembered_move
    move, score, ponder, remembered_move = turk.bestMove(board, remembered_move)
    return "bestmove " + str(move) + " ponder " + str(ponder)


def handle_uci_input(line):
    if line.startswith("position"):
        parse_position(line)
    elif line.startswith("go"):
        return handle_go(line)
    elif line == "uci":
        return "uciok"
    elif line == "isready":
        return "readyok"
    return line


def main():
    stdin = NonBlockingStreamReader(sys.stdin)
    while True:
        line = stdin.readline(0.1)
        if line != None:
            print(handle_uci_input(line.strip()))


if __name__ == '__main__':
    main()
