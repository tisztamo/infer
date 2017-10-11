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
    # global candidate_moves
    # for candidate in candidate_moves:
    #     if candidate != None:
    #         candidate["uci"] = None
    #         candidate["prob"] = 0
    #         candidate["appeal"] = 0
    # words = line.split(" ")
    # if PLAY_AGAINST_HUMAN:
    #     retval = "go movetime 1000"
    # else:
    #     if len(words) <= 1 or words[1] != "wtime":
    #         return line
    #     try:
    #         if words[5] == "winc":
    #             words[6] = str(0)
    #             words[8] = str(0)
    #     except:
    #         pass
    #     if i_am_white():
    #         mytime_idx = 2
    #     else:
    #         mytime_idx = 4
    #     mytime = int(words[mytime_idx])
    #     modded_time = mytime
    #     if board.halfmove_clock < 30:
    #         modded_time = modded_time * (board.halfmove_clock / 60 + 0.5)
    #     modded_time = modded_time * 0.8
    #     if modded_time > 30000:
    #         modded_time = 30000
    #     words[mytime_idx] = str(int(modded_time))
    #     retval = " ".join(words)
    # return retval

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

# def target_loss(best_score):
#     if best_score < DESIRED_ADVANTAGE:
#         target = 0
#     else:
#         target = (best_score - DESIRED_ADVANTAGE) * 0.15
#     target = max(min_target_loss, min(target, MAX_LOSS_PER_MOVE))
#     print("Best Score:", best_score, "Target loss:", target)
#     return target

# def print_candidate_moves(candidate_moves):
#     moves = sorted(candidate_moves, key=lambda c: -c["prob"])
#     for move in moves:
#         try:
#             print("Candidate", move["uci"], move["prob"], move["lost_score"], move["appeal"])
#         except:
#             print(move)

# def override_bestmove(best_move):
#     global board
#     global predictions
#     global candidate_moves
#     my_best = None
#     lost_score = 0

#     best_score = candidate_moves[0]["score"]
#     target = target_loss(best_score)
#     if best_score == None:
#         print("No score was provided for the best candidate")
#         print(candidate_moves)
#         my_best = candidate_moves[0]
#         my_best["prob"] = -1
#     else:
#         for candidate in candidate_moves:
#             uci = candidate["uci"]
#             if uci == None:
#                 continue
#             candidate["prob"] = -1
#             try:
#                 move_loss = best_score - candidate["score"]
#                 idx = inference.label_strings.index(uci)
#                 probability = predictions[idx]
#             except:
#                 probability = 0
#                 move_loss = 22222
#             candidate["lost_score"] = move_loss
#             candidate["prob"] = round(probability * 10)
#             candidate["appeal"] = round(
#                 probability / (abs(move_loss - target) * 0.05 + 5) * 1000)

#         candidate_moves.sort(key=lambda c: -c["appeal"])

#         my_best = best_move
#         for candidate in candidate_moves:
#             if candidate["lost_score"] < MAX_LOSS_PER_MOVE:
#                 my_best = candidate
#                 lost_score = candidate["lost_score"]
#                 break

#     my_best["label"] = board.san(chess.Move.from_uci(my_best["uci"]))

#     print_candidate_moves(candidate_moves)
#     if lost_score == 0:
#         print("Selected best:", my_best)
#     else:
#         print("Selected:", my_best, "and Lost", lost_score, "centipawns. Target was", target)
    
#     print("Move:", my_best["uci"], target, lost_score, my_best.get("prob", 0))
#     return my_best["uci"]

def main():
    stdin = NonBlockingStreamReader(sys.stdin)
    while True:
        line = stdin.readline(0.1)
        if line != None:
            print(handle_uci_input(line.strip()))


if __name__ == '__main__':
    main()
