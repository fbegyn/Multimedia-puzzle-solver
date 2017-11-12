import cv2
import numpy as np

class puzzleSolver:
    # puzzleSolver(puzzles [], type)
    def __init__(self, puzzles):
        self.puzzle = puzzles
        self.__orb = cv2.ORB_create()
        self.__bf= cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def handle_pieces(self):
        pieces = self.puzzle.pieces
        for i in pieces:
            kp1, des1 = self.__orb.detectAndCompute(i,None)
            for j in pieces:
                kp2, des2 = self.__orb.detectAndCompute(j,None)
                matches = self.__bf.match(des1, des2)
                matches = sorted(matches, key=lambda x:x.distance)
                output = cv2.drawMatches(i, kp1, j, kp2, matches[:5], None, flags=2)
                cv2.imshow('matched', output)
                cv2.waitKey(1000)
                cv2.destroyWindow('matched')


