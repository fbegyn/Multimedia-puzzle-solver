import cv2
import numpy as np

class Puzzle:
    # Initialisation for the puzzle class. Pass a path to the class contructor
    # and it will amek sure it can do everything correctly
    def __init__(self, path):
        self.puzzle = cv2.imread(path)
        self.gray = cv2.cvtColor(self.puzzle, cv2.COLOR_BGR2GRAY)

    # show the original puzzle solution
    def show(self, time=0):
        cv2.imshow('Puzzle', self.puzzle)
        cv2.waitKey(time)
        cv2.destroyWindow('Puzzle')

    # Return the contours if they exist
    def get_contours(self):
        if self.contours is None:
            print('Contours don\'t exist yet, have you used Puzzle.contours()')
        else:
            return self.__contours, self.__hierarchy

    # Return the contours a a vector of all points and the hierachy of the contours
    def contours(self):
        self.__thresh = np.asarray((self.gray>0)*255,dtype= np.uint8)
        _, contours, hierarch = cv2.findContours(self.__thresh.copy(), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_NONE)
        self.__contours = contours
        self.__hierarchy = hierarch

    # Show the contours
    def draw_contours(self, time=0):
        if self.__contours is None:
            self.contours()
        # Draw contours
        h, w = self.__thresh.shape[:2]
        bg = self.puzzle.copy()
        for i in self.__contours:
            cv2.drawContours(bg, i, -1, (128,255,255), -1)
        cv2.imshow('contours', bg)
        cv2.waitKey(time)
        cv2.destroyWindow('contours')

    # Given the contours, determine all the puzzle pieces
    def calc_pieces(self, margin=25, draw=False):
        self.pieces = []
        offset = int(margin/2)
        if draw:
            mask = self.puzzle.copy()

        if self.__contours is None:
            self.contours()

        h, w = self.__thresh.shape[:2]
        for i in self.__contours:
            rect = cv2.boundingRect(i)
            x, y, w, h = rect
            self.pieces.append(self.puzzle[y-offset:y+h+offset,x-offset:x+w+offset].copy())
            if draw:
                cv2.rectangle(mask, (x-offset, y-offset), (x+w+offset, y+h+offset), (0, 255, 0), 1)

        if draw:
            cv2.imshow('test', mask)
            cv2.waitKey()
            cv2.destroyWindow('test')

    def show_pieces(self, time=500):
        if self.pieces is None:
            self.calc_pieces()

        for p in self.pieces:
            cv2.imshow('test', p)
            cv2.waitKey(time)
            cv2.destroyWindow('test')

