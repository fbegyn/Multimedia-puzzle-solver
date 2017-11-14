import lib.puzzle as puzzle
import lib.puzzleSolver as puzzleSolver


def show(img, time=0):
        cv2.imshow('Puzzle',img)
        cv2.waitKey(time)
        cv2.destroyWindow('Puzzle')


puzz1 = puzzle.Puzzle('../puzzles/Tiles/tiles_rotated/tiles_rotated_5x5_02.png')
#puzz1.show()

#puzz1.contours()
#puzz1.draw_contours()
#puzz1.calc_pieces(margin=8, draw=True)
#puzz1.show_pieces()


solver = puzzleSolver.puzzleSolver(puzz1)
solver.slice_image(3)
#p = solver.puzzle.pieces

#show(p[0])


solver.puzzle.show_pieces()
