from __future__ import print_function

import sys, time
import fnmatch
import engine
import numpy as np
import tensorflow as tf
import chess
import log
import pgnwriter

logger = log.getLogger("uci")

board = chess.Board()
turk = engine.Engine()
pgn_writer = pgnwriter.PGNWriter(turk)

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
    if not found_move:
        logger.info("Transition move not found from " + from_board.fen() + " to " + to_board.fen())
    return found_move


def parse_position(line):
    global board
    words = line.split(" ")
    if words[1] == "startpos":
        board = chess.Board()
        for move in words[3:]:
            try:
                board.push_uci(move)
            except Exception as e:
                logger.error(e)
    elif words[1] == "fen":
        fen = " ".join(words[2:]).replace(" -1 ", " - ")
        board = chess.Board(fen)
        move = find_transition_move(turk.current_board, board)
        if move:
            board = turk.current_board.copy()
            board.push(move)
            pgn_writer.move(move)
    print(board.fen())

#go wtime 39360 btime 38640 movestogo 34
def parse_go(line):
    logger.info(line)
    params = {}
    words = line.split(" ")
    key = None
    for word in words[1:]:
        if key is None:
            key = word
        else:
            params[key] = word
            key = None
    return params

def handle_go(line):
    params = parse_go(line)
    move = turk.move(board, params)
    pgn_writer.move(move)
    ponder = move.ponder if move.ponder is not None else "a1a2"
    return "bestmove " + str(move.uci) + " ponder " + str(ponder)


def handle_uci_input(line):
    if line.startswith("position"):
        parse_position(line)
    elif line.startswith("go"):
        return handle_go(line)
    elif line == "uci":
        name = turk.name
        return "id name " + name + "\n" + "uciok"
    elif line == "ucinewgame":
        turk.newGame()
        pgn_writer.new_game()
    elif line == "isready":
        return "readyok"
    elif line == "quit":
        pgn_writer.end_game()
        sys.exit(0)
    return line

# def target_loss(best_score):
#     if best_score < DESIRED_ADVANTAGE:
#         target = 0
#     else:
#         target = (best_score - DESIRED_ADVANTAGE) * 0.15
#     target = max(min_target_loss, min(target, MAX_LOSS_PER_MOVE))
#     print("Best Score:", best_score, "Target loss:", target)
#     return target


def main(unused_argv):
    turk.initBackEngine()
    while True:
        line = sys.stdin.readline()
        if line != None:
            print(handle_uci_input(line.strip()))


if __name__ == '__main__':
    tf.app.run()
