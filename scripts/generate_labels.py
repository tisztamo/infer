rows = ["a", "b", "c", "d", "e", "f", "g", "h"]
columns = [1, 2, 3, 4, 5, 6, 7, 8]

def ucicoord(columnidx, rowidx):
  return rows[rowidx] + str(columns[columnidx])


def generate_piece(from_c, from_r, directions, max_distance, promote_to = ""):
  from_coords = ucicoord(from_c, from_r)
  moves = []
  for direction in directions:
    for length in range(1, max_distance + 1):
      column = from_c + direction[0] * length
      row = from_r + direction[1] * length
      if column in range(0, 8) and row in range(0, 8):
        moves.append(from_coords + ucicoord(column, row) + promote_to)
  return moves

def generate_queen(from_c, from_r):
  directions = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]
  return generate_piece(from_c, from_r, directions, 7)

def generate_knight(from_c, from_r):
  directions = [[-1, 2], [-1, -2], [1, 2], [1, -2], [-2, -1], [-2, 1], [2, -1], [2, 1]]
  return generate_piece(from_c, from_r, directions, 1)

def generate_pawn_promotions(from_c, from_r):
  if from_c == 1:
    directions = [[-1, -1], [-1, 0], [-1, 1]]
  else:
    directions = [[1, -1], [1, 0], [1, 1]]
  moves = []
  for promote_to in ["n", "b", "r", "q"]:
    moves.extend(generate_piece(from_c, from_r, directions, 1, promote_to))
  return moves

def generate_castlings():
  return ["e1g1", "e1c1", "e8g8", "e8c8"]

moves = []
for columnidx in range(len(columns)):
  for rowidx in range(len(rows)):
    moves.extend(generate_queen(columnidx, rowidx))
    moves.extend(generate_knight(columnidx, rowidx))
    if columnidx in [1, 6]:
      moves.extend(generate_pawn_promotions(columnidx, rowidx))

moves.extend(generate_castlings())

moves.sort()
print(" ".join(moves))