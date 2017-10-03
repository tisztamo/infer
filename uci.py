from __future__ import print_function

import sys
import fnmatch
import chess
from subprocess import Popen, PIPE, STDOUT
from nbstreamreader import NonBlockingStreamReader
import inference
import numpy as np

PLAY_AGAINST_HUMAN = True
MULTIPV = 10
MAX_LOSS_PER_MOVE = 100

board = chess.Board()
predictions = None
candidate_moves = [None] * MULTIPV


def send_options(engine):
    engine.stdin.write("setoption name MultiPV value " + str(MULTIPV) + "\n")
#    engine.stdin.write("setoption name Slow Mover value 20\n")
    engine.stdin.flush()


def open_engine():
    engine = Popen("../stockfish-8-linux/Linux/stockfish_8_x64_modern",
                   stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    send_options(engine)
    outputStream = NonBlockingStreamReader(engine.stdout)
    return engine, outputStream


def i_am_white():
    return board.turn == chess.WHITE


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
    global candidate_moves
    for candidate in candidate_moves:
        if candidate != None:
            candidate["uci"] = None
            candidate["prob"] = 0
            candidate["appeal"] = 0
    words = line.split(" ")
    if PLAY_AGAINST_HUMAN:
        retval = "go movetime 4000"
    else:
        if len(words) <= 1 or words[1] != "wtime":
            return line
        try:
            if words[5] == "winc":
                words[6] = str(0)
                words[8] = str(0)
        except:
            pass
        if i_am_white():
            mytime_idx = 2
        else:
            mytime_idx = 4
        mytime = int(words[mytime_idx])
        modded_time = mytime
        if board.halfmove_clock < 30:
            modded_time = modded_time * (board.halfmove_clock / 60 + 0.5)
        modded_time = modded_time * 0.8
        if modded_time > 30000:
            modded_time = 30000
        words[mytime_idx] = str(int(modded_time))
        retval = " ".join(words)
    return retval


def handle_uci_input(line):
    global predictions
    if line.startswith("position"):
        parse_position(line)
        predictions = inference.predict(board.fen())
    elif line.startswith("setoption name MultiPV"):
        return False
    elif line.startswith("go"):
        return handle_go(line)
    return line

def target_loss(best_score):
    return 0
    if best_score < 20:
        target = 0
    else:
        target = (best_score - 20) * 0.2
    target = min(target, MAX_LOSS_PER_MOVE)
    print("Best Score:", best_score, "Target loss:", target)
    return target

def override_bestmove(best_move):
    global board
    global predictions
    global candidate_moves
    my_best = None
    lost_score = 0

    best_score = candidate_moves[0]["score"]
    target = target_loss(best_score)
    if best_score == None:
        print("No score was provided for the best candidate")
        print(candidate_moves)
        my_best = candidate_moves[0]
        my_best["prob"] = -1
    else:
        for candidate in candidate_moves:
            uci = candidate["uci"]
            if uci == None:
                continue
            candidate["prob"] = -1
            try:
                move_loss = best_score - candidate["score"]
                idx = inference.label_strings.index(uci)
                probability = predictions[idx]
            except:
                probability = 0
                move_loss = 22222
            candidate["lost_score"] = move_loss
            candidate["prob"] = round(probability * 10)
            candidate["appeal"] = round(
                probability / (abs(move_loss - target) * 0.1 + 5) * 1000)

        candidate_moves.sort(key=lambda c: -c["appeal"])

        my_best = best_move
        for candidate in candidate_moves:
            if candidate["lost_score"] < MAX_LOSS_PER_MOVE:
                my_best = candidate
                lost_score = candidate["lost_score"]
                break

    my_best["label"] = board.san(chess.Move.from_uci(my_best["uci"]))

    print(candidate_moves)
    if lost_score == 0:
        print("Selected best:", my_best)
    else:
        print("Selected:", my_best, "and Lost", lost_score, "centipawns. Target was", target)
    
    print("Move:", my_best["uci"], target, lost_score, my_best.get("prob", 0))
    return my_best["uci"]


def parse_multipv(line):
    global candidate_moves
    lastword = ""
    pv = None
    multipv = None
    score = None
    for word in line.split(" "):
        if lastword == "multipv":
            multipv = int(word)
        elif lastword == "pv":
            pv = word
        elif lastword == "cp":
            score = int(word)
        lastword = word
    if pv is None or multipv is None:
        return False
    else:
        idx = multipv - 1
        candidate_moves[idx] = {"score": score, "uci": pv}


def handle_uci_from_engine(line):
    global predictions
    global candidate_moves
    if line.startswith("bestmove"):
        move = override_bestmove(line.split(" ")[1])
        print("bestmove", move)
        return False
    if line.startswith("info depth"):
        parse_multipv(line)
        return False
    return True


def main():
    stdin = NonBlockingStreamReader(sys.stdin)
    board = chess.Board()
    movements = ""
    engine, engine_output = open_engine()
    while True:
        while True:
            line = engine_output.readline(0.1)
            if line != None:
                if handle_uci_from_engine(line.strip()):
                    print(line.strip())
            else:
                break
        while True:
            line = stdin.readline(0.1)
            if line != None:
                modded_line = handle_uci_input(line.strip())
                if modded_line:
                    engine.stdin.write(modded_line)
                    engine.stdin.write("\n")
                    engine.stdin.flush()
                    print("PASSED:", modded_line)
            else:
                break


if __name__ == '__main__':
    main()
