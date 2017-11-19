import cv2
import numpy as np

class Puzzle:
    # Initialisation for the puzzle class. Pass a path to the class contructor
    # and it will amek sure it can do everything correctly
    def __init__(self, path):
        self.puzzle = cv2.imread(path)
        self.gray = cv2.cvtColor(self.puzzle, cv2.COLOR_BGR2GRAY)
        self.dimh=0 #puzzle slicing parameters
        self.dimv=0
        self.piece_h = 0 #piece dimentions in pixels
        self.piece_v = 0
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

        width, height = self.puzzle.shape[:2]

        if draw:
            mask = self.puzzle.copy()

        if self.__contours is None:
            self.contours()

        for i in self.__contours:
            _, _, angle = cv2.minAreaRect(i)
            rect = cv2.boundingRect(i)
            x, y, w, h = rect
            y_start = y-offset
            x_start = x-offset
            y_end = y+h+offset
            x_end = x+w+offset
            if y_start < 0:
                y_start = 0
            if x_start < 0:
                x_start = 0
            if y_end > height:
                y_end = height-1
            if x_end > width:
                x_end = width-1

            piece = self.puzzle[y_start:y_end,x_start:x_end].copy()

            cols, rows = piece.shape[:2]
            rotM = cv2.getRotationMatrix2D((cols/2, rows/2), int(angle), 1)
            piece = cv2.warpAffine(piece, rotM, (cols, rows))

            self.pieces.append(piece)
            if draw:
                cv2.rectangle(mask, (x_start, y_start), (x_end, y_end), (0, 255, 0), 1)

        if draw:
            cv2.imshow('test', mask)
            cv2.waitKey()
            cv2.destroyWindow('test')

        print('Found {} puzzle pieces.'.format(len(self.pieces)))

    def show_pieces(self, time=500):
        if self.pieces is None:
            self.calc_pieces()

        for p in self.pieces:
            cv2.imshow('test', p)
            cv2.waitKey(time)
            cv2.destroyWindow('test')

