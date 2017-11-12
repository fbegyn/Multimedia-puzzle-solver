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
    def calc_pieces(self):
        if self.__contours is None:
            self.contours()


        print(self.__contours)
        print('----------------')
        h, w = self.__thresh.shape[:2]
        for i in self.__contours:
            print(np.amax(i))
            print(np.amin(i))
            print('------------------')
            mask = np.zeros_like(self.puzzle)
            cv2.drawContours(mask, i, -1, (255, 255, 255), -1)
            out = np.zeros_like(mask)
            out[mask==255] = self.puzzle[mask==255]
            cv2.imshow('test', out)
            cv2.waitKey()
            cv2.destroyWindow('test')


        return None
