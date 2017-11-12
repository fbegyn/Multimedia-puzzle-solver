import lib.puzzle as puzzle
import lib.puzzleSolver as puzzleSolver

puzz1 = puzzle.Puzzle('puzzles/Tiles/tiles_scrambled/tiles_scrambled_5x5_08.png')
puzz1.show()

puzz1.contours()
puzz1.draw_contours()
puzz1.calc_pieces(margin=8, draw=True)
puzz1.show_pieces()

solver = puzzleSolver.puzzleSolver(puzz1)
