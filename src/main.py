import lib.puzzle as puzzle
import lib.puzzleSolver as puzzleSolver
import numpy as np

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


im1 = 3
im2 = 0

n1, e1, s1, w1 = solver.give_edges_NESW(im1)
n2, e2, s2, w2 = solver.give_edges_NESW(im2)
img = np.hstack([solver.puzzle.pieces[im1], solver.puzzle.pieces[im2]])



weight = puzzleSolver.puzzleSolver.compare_rgb_slices(e1,w2)
print(weight)

show(img)



#solver.puzzle.show_pieces()
