import collections
from datetime import date
import chess, chess.pgn

MY_NAME = "Turk Development"


def board_to_game(board):
    game = chess.pgn.Game()

    # Undo all moves.
    switchyard = collections.deque()
    while board.move_stack:
        switchyard.append(board.pop())

    game.setup(board)
    node = game

    # Replay all moves.
    while switchyard:
        move = switchyard.pop()
        node = node.add_variation(move)
        board.push(move)

    game.headers["Result"] = board.result()
    return game

def write_history(board, engine=None):
    if board.fullmove_number > 1:
        game = board_to_game(board)
        if engine:
            if engine.color == chess.WHITE:
                game.headers["White"] = engine.name
            else:
                game.headers["Black"] = engine.name
            game.headers["Date"] = date.today().isoformat().replace("-", ".")
        with open("history.pgn", "a") as pgn_file:
            exporter = chess.pgn.FileExporter(pgn_file)
            game.accept(exporter)