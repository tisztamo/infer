from __future__ import print_function
import sys
import chess, chess.uci, chess.pgn

# Usage: predictiontest.py engine_exe pgn_file [player]

WHITE = 1
BLACK = 2

def next_game(input_file, player_name):
    game = None
    player_color_mask = None
    while game is None:
        game = chess.pgn.read_game(input_file)
        if game is None:
            break
        player_color_mask = WHITE | BLACK
        if player is not None:
            if game.headers["White"] == player:                        
                player_color_mask = WHITE
            elif game.headers["Black"] == player:                        
                player_color_mask = BLACK
            else:
                print(player + " not found in game " + game.headers["White"] + " vs " + game.headers["Black"])
                game = None
                continue
    return game, player_color_mask


def turn_to_colormask(chess_turn_bool):
    """Converts pychess style True/False to WHITE or BLACK"""
    return WHITE if chess_turn_bool else BLACK

def analyse_game(game, engine, player_color_mask):
    board = game.board()
    engine.ucinewgame()
    ply_count = 0
    correct_pred_count = 0
    for move in game.main_line():
        if turn_to_colormask(board.turn) & player_color_mask != 0:
            ply_count += 1
            engine.position(chess.Board(board.fen()))#Previous moves will not be sent!
            engine_move, _ = engine.go(depth=18)
            if engine_move.uci() == move.uci():
                correct_pred_count += 1
            print(str(correct_pred_count) + "/" + str(ply_count), end="\r")
        board.push(move)
    return ply_count, correct_pred_count

def process_file(pgn_file_name, engine, player_name=None):
    game = None
    total_ply_count = 0
    total_correct_pred_count = 0
    with open(pgn_file_name) as input_file:
        while True:
            game, player_color_mask = next_game(input_file, player_name)
            if game is None:
                break
            ply_count, correct_pred_count = analyse_game(game, engine, player_color_mask)
            total_correct_pred_count += correct_pred_count
            total_ply_count += ply_count
            print(game.headers["White"] + " vs " + game.headers["Black"] + ": %.2f%%, "  % (float(correct_pred_count) / float(ply_count) * 100.0) + str(correct_pred_count) + " / " + str(ply_count))
    print("Prediction accuracy: %.2f%% (%d from %d)" % (float(total_correct_pred_count) / float(total_ply_count) * 100.0, total_correct_pred_count, total_ply_count))
    return total_correct_pred_count, total_ply_count           

if __name__ == "__main__":
    engine_exe = sys.argv[1]
    pgn_file = sys.argv[2]
    player = None
    if len(sys.argv) > 3:
        player = sys.argv[3]
    print("Engine exe: " + engine_exe + " pgn_file: " + pgn_file + " player: " + str(player))
    engine = chess.uci.popen_engine(engine_exe)
    engine.uci()
    process_file(pgn_file, engine, player)
    engine.quit()
