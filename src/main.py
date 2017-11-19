import lib.puzzle as puzzle
import lib.puzzleSolver as puzzleSolver
import numpy as np
import cv2


def show(img, time=0):
        cv2.imshow('Puzzle',img)
        cv2.waitKey(time)
        cv2.destroyWindow('Puzzle')

puzz1 = puzzle.Puzzle('../puzzles/Tiles/tiles_shuffled/tiles_shuffled_2x3_02.png')
#puzz1.show()

#puzz1.contours()
#puzz1.draw_contours()
#puzz1.calc_pieces(margin=8, draw=True)
#puzz1.show_pieces()


solver = puzzleSolver.puzzleSolver(puzz1)
solver.slice_image(2,3)
mapper  = solver.get_mapper()
match = solver.get_best_match_from_mapper(mapper)
solution = solver.get_solution_from_best_match(match)












