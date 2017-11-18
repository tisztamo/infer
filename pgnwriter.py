import collections
from datetime import date
import chess, chess.pgn
import engine

MY_NAME = "Turk Development"

class PGNWriter():
    def __init__(self, engine=None, file_name="history.pgn"):
        self.board = None
        self.game = None
        self.last_node = None
        self.engine = engine
        self.file_name = file_name
        self.new_game()

    def new_game(self):
        if self.board is not None and self.board.fullmove_number > 1:
            self.end_game()
        self.board = chess.Board()
        self.game = chess.pgn.Game()
        self.last_node = self.game

    def end_game(self, result = None):
        if result is None:
            result = self.board.result()
        self.game.headers["Result"] = result
        self.write_history()

    def move(self, move_):
        lost_score = None
        if isinstance(move_, str):
            move = chess.Move.from_uci(move_)
        elif isinstance(move_, engine.CandidateMove):
            move = chess.Move.from_uci(move_.uci) if move_.uci is not None else None
            lost_score = move_.lost_score
        else:
            move = move_
        if move is not None:
            self.last_node = self.last_node.add_variation(move)
            if lost_score is not None:
                if lost_score >= 300:
                    self.last_node.nags.add(chess.pgn.NAG_BLUNDER)
                elif lost_score >= 100:
                    self.last_node.nags.add(chess.pgn.NAG_MISTAKE)
            self.board.push(move)
        if self.board.is_game_over():
            self.end_game()

    def write_history(self):
        if self.engine:
            if self.engine.color == chess.WHITE:
                self.game.headers["White"] = self.engine.name
            else:
                self.game.headers["Black"] = self.engine.name
            self.game.headers["Date"] = date.today().isoformat().replace("-", ".")
        with open(self.file_name, "a") as pgn_file:
            exporter = chess.pgn.FileExporter(pgn_file)
            try:
                self.game.accept(exporter)
            except Exception as e:
                print("Unable to export game", e)