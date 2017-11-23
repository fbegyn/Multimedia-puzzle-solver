import lib.puzzle as puzzle
import lib.puzzleSolver as puzzleSolver
import numpy as np
import cv2
from timeit import default_timer as timer

def show(img, time=0):
        cv2.imshow('Puzzle',img)
        cv2.waitKey(time)
        cv2.destroyWindow('Puzzle')

puzz0 = puzzle.Puzzle('../puzzles/Tiles/tiles_rotated/tiles_rotated_3x3_00.png')
puzz1 = puzzle.Puzzle('../puzzles/Tiles/tiles_rotated/tiles_rotated_3x3_01.png')
puzz2 = puzzle.Puzzle('../puzzles/Tiles/tiles_rotated/tiles_rotated_3x3_02.png')
puzz3 = puzzle.Puzzle('../puzzles/Tiles/tiles_rotated/tiles_rotated_3x3_03.png')
puzz4 = puzzle.Puzzle('../puzzles/Tiles/tiles_rotated/tiles_rotated_3x3_04.png')
puzz5 = puzzle.Puzzle('../puzzles/Tiles/tiles_rotated/tiles_rotated_3x3_05.png')
puzz6 = puzzle.Puzzle('../puzzles/Tiles/tiles_rotated/tiles_rotated_3x3_06.png')
puzz7 = puzzle.Puzzle('../puzzles/Tiles/tiles_rotated/tiles_rotated_3x3_07.png')
puzz8 = puzzle.Puzzle('../puzzles/Tiles/tiles_rotated/tiles_rotated_3x3_08.png')

puzz = np.array([puzz0, puzz1, puzz2, puzz3, puzz4, puzz5, puzz6, puzz7, puzz8])

avg_time = 0

#len(puzz)

#2,3
#7,8
#2,5
for i in range(len(puzz)):
    time = timer()
    solver = puzzleSolver.puzzleSolver(puzz[i])
    solver.slice_image(3,3)
    mapper  = solver.get_mapper()
    match = solver.get_best_match_from_mapper(mapper)
    solution = solver.get_solution_from_best_match(match)
    timing = timer() - time
    avg_time +=timing
    print(timing)
    
    #comment out show command for accurat timing info.
    #show(solution)

avg_time /= len(puzz)
print("")
print("average execution time")
print(avg_time)


