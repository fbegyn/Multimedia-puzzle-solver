import lib.puzzle as puzzle
import lib.puzzleSolver as puzzleSolver

puzz1 = puzzle.Puzzle('puzzles/jigsaw/jigsaw_scrambled/jigsaw_scrambled_3x3_01.png')
puzz1.show()


puzz1.contours()
puzz1.draw_contours()
puzz1.calc_pieces()
