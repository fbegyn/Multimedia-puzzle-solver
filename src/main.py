import lib.puzzle as puzzle
import lib.puzzleSolver as puzzleSolver

puzz1 = puzzle.Puzzle('puzzles/Tiles/tiles_scrambled/tiles_scrambled_5x5_04.png')
puzz1.show()


puzz1.contours()
puzz1.calc_pieces(margin=10, draw=True)
