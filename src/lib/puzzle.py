import cv2

class Puzzle:
    puzzle = None
    gray = None
    def __init__(self, path):
        self.puzzle = cv2.imread(path)
        gray = cv2.cvtColor(self.puzzle, cv2.COLOR_BGR2GRAY)

    def show(self, time=None):
        cv2.imshow('Puzzle', self.puzzle)
        cv2.waitKey(time)
        cv2.destroyWindow('Puzzle')

    def get(self):
        return selfpuzzle

    def contours(self):

