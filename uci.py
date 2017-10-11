from __future__ import print_function

import sys, time
import fnmatch
from nbstreamreader import NonBlockingStreamReader
import engine
import numpy as np
import chess
import log

logger = log.getLogger("uci")

board = chess.Board()
turk = engine.Engine()

def find_transition_move(from_board, to_board):
    try_board = from_board.copy()
    to_fen = to_board.board_fen()
    found_move = None
    for move in from_board.legal_moves:
        try_board.push(move)
        if try_board.board_fen() == to_fen:
            found_move = move
            break
        try_board.pop()
    if not move:
        logger.log("Transition move not found from " + from_board.fen() + " to " + to_board.fen())
    return found_move


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
        move = find_transition_move(turk.current_board, board)
        if move:
            board = turk.current_board.copy()
            board.push(move)


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
