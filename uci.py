from __future__ import print_function

import sys, time
import fnmatch
from nbstreamreader import NonBlockingStreamReader
import engine
import numpy as np
import chess

board = chess.Board()
turk = engine.Engine()

def parse_position(line):
    global board
    words = line.split(" ")
    if words[1] == "startpos":
        board = chess.Board()
        for move in words[3:]:
            board.push_uci(move)
    elif words[1] == "fen":
        fen = " ".join(words[2:]).replace(" -1 ", " - ")
        board = chess.Board(fen)


def handle_go(line):
    move, score, ponder = turk.move(board)
    return "bestmove " + str(move) + " ponder " + str(ponder)


def handle_uci_input(line):
    if line.startswith("position"):
        parse_position(line)
    elif line.startswith("go"):
        return handle_go(line)
    elif line == "uci":
        return "uciok"
    elif line == "ucinewgame":
        turk.newGame()
    elif line == "isready":
        return "readyok"
    elif line == "quit":
        turk.newGame()
        sys.exit(0)
    return line


def main():
    stdin = NonBlockingStreamReader(sys.stdin)
    while True:
        line = stdin.readline(0.1)
        if line != None:
            print(handle_uci_input(line.strip()))


if __name__ == '__main__':
    main()
