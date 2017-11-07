import cv2
import numpy as np

class Puzzle:
    puzzle = None
    gray = None
    def __init__(self, path):
        self.puzzle = cv2.imread(path)
        self.gray = cv2.cvtColor(self.puzzle, cv2.COLOR_BGR2GRAY)

    def show(self, time=0):
        cv2.imshow('Puzzle', self.puzzle)
        cv2.waitKey(time)
        cv2.destroyWindow('Puzzle')

    def get(self):
        return self.puzzle

    def contours(self):
        tresh = np.asarray((self.gray>0)*255,dtype= np.uint8)
        _, contours, hierarch = cv2.findContours(tresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        print(contours[0])
        # Draw contours
        h, w = tresh.shape[:2]
        vis = np.zeros((h, w, 3), np.uint8)
        cv2.drawContours( vis, contours[0], -1, (128,255,255), -1)
        cv2.imshow('cont', vis)
        cv2.waitKey()
