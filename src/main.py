import lib.puzzle as puzzle
import lib.puzzleSolver as puzzleSolver
import numpy as np
import cv2


def show(img, time=0):
        cv2.imshow('Puzzle',img)
        cv2.waitKey(time)
        cv2.destroyWindow('Puzzle')

puzz0 = puzzle.Puzzle('../puzzles/Tiles/tiles_shuffled/tiles_shuffled_3x3_00.png')
puzz1 = puzzle.Puzzle('../puzzles/Tiles/tiles_shuffled/tiles_shuffled_3x3_01.png')
puzz2 = puzzle.Puzzle('../puzzles/Tiles/tiles_shuffled/tiles_shuffled_3x3_02.png')
puzz3 = puzzle.Puzzle('../puzzles/Tiles/tiles_shuffled/tiles_shuffled_3x3_03.png')
puzz4 = puzzle.Puzzle('../puzzles/Tiles/tiles_shuffled/tiles_shuffled_3x3_04.png')
puzz5 = puzzle.Puzzle('../puzzles/Tiles/tiles_shuffled/tiles_shuffled_3x3_05.png')
puzz6 = puzzle.Puzzle('../puzzles/Tiles/tiles_shuffled/tiles_shuffled_3x3_06.png')
puzz7 = puzzle.Puzzle('../puzzles/Tiles/tiles_shuffled/tiles_shuffled_3x3_07.png')
puzz8 = puzzle.Puzzle('../puzzles/Tiles/tiles_shuffled/tiles_shuffled_3x3_08.png')


puzz = np.array([puzz0, puzz1, puzz2, puzz3, puzz4, puzz5, puzz6, puzz7, puzz8])
best_start = np.array([6, 8, 8, 5, 8, 6, 7, 4 ,2])

for i in range(len(puzz)):
    solver = puzzleSolver.puzzleSolver(puzz[i])
    solver.slice_image(3,3)
    mapper  = solver.get_mapper()
    match = solver.get_best_match_from_mapper(mapper, best_start[i])
    solution = solver.get_solution_from_best_match(match)

    cv2.imshow('Solution',solution)
    cv2.waitKey(0)
    cv2.destroyWindow('Solution')

