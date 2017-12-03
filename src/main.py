


import lib.puzzle as puzzle
import lib.puzzleSolver as puzzleSolver
import numpy as np
import cv2
import glob
from timeit import default_timer as timer
import matplotlib.pyplot as plt

def show(img, title='Puzzle', time=0):
    cv2.imshow(title,img)
    cv2.waitKey(time)
    cv2.destroyWindow(title)

puzzle_list= []
folder = "../puzzles/Tiles/tiles_rotated/"
for path in sorted(glob.glob(folder + '*.png')):
    #print(path)
    filename = path.split("/")[4]
    dim = filename.split("_")[2].split("x")
    print(filename.split(".")[0].split("_")[3])
    puzzle_list.append([puzzle.Puzzle(path), int(dim[0]), int(dim[1]), filename])


#[puzzle_list[36]]
'''
time = timer()
for i in range(10):
    for p in puzzle_list:
        solver = puzzleSolver.puzzleSolver(p[0])
        solver.slice_image(p[1],p[2])
        mapper  = solver.get_mapper()
        match = solver.get_best_match_from_mapper(mapper)
        solution = solver.get_solution_from_best_match(match)
        #comment out show command for accurat timing info.
        #show(solution, title=p[3])
timing = (timer() - time)/10
print("Total execution time:")
print(np.ceil(timing*1000)/1000, "sec")


'''











