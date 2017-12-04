import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import chess
import inference, input

app = Flask(__name__)
CORS(app)

label_strings, _ = input.load_labels()

@app.route("/intuition", methods=["GET"])
def intuition():
    fen = request.args.get('fen', None)
    board = chess.Board(fen)
    moves = request.args.get('moves', None)
    if moves is not None:
        for move in moves.split(" "):
            try:
                board.push_uci(move)
            except Exception as e:
                print(e)
    predictions = inference.predict_move(board.fen())
    candidates = np.argpartition(predictions, -20)[-20:]
    candidates = candidates[np.argsort(-predictions[candidates])]
    retval = []
    for i, candidate in enumerate(candidates):
        prob = float(predictions[candidate])
        if prob > 0.01:
            retval.append({"move": label_strings[candidate], "prob": prob})
    return jsonify(retval)

app.run(host= '0.0.0.0')
