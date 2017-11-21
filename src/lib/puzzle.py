import cv2
import numpy as np

class Puzzle:
    def __init__(self, path):
        """ Initialisation function for the puzzle class,
            Stores the picture and a graysclae version of it """
        self.puzzle = cv2.imread(path)
        self.gray = cv2.cvtColor(self.puzzle, cv2.COLOR_BGR2GRAY)

    def show(self, time=0):
        """ Shows the picture itself """
        cv2.imshow('Puzzle', self.puzzle)
        cv2.waitKey(time)
        cv2.destroyWindow('Puzzle')

    def get_contours(self):
        """ Fetches the contours if they exist, otherwise returns an error info code """
        if self.contours is None:
            print('Contours don\'t exist yet, have you used Puzzle.contours()')
        else:
            return self.__contours, self.__hierarchy
    
    # TODO: Do reseqrch into the algorithm that findcontours uses
    def contours(self):
        """ Determines the contours of the pictures based on the opencv findContours
            function """
        self.__thresh = np.asarray((self.gray>0)*255,dtype= np.uint8)
        _, contours, hierarch = cv2.findContours(self.__thresh.copy(), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_NONE)
        self.__contours = contours
        self.__hierarchy = hierarch


    def draw_contours(self, time=0):
        """ Draw the contours on top of the picture """
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

    #TODO: Move the code from the puzzlesolver in here, the puzzlesolver doesn't need to modify
    # The picturs in this way, that's the job of the puzzle itself
    def slice_image(self, parts):
        slices = []

        size_y = int(len(self.puzzle)/parts)
        size_x = int(len(self.puzzle[0])/parts)

        for i in range(parts):
            slices.append(self.puzzle[i*size_y:(i+1)*size_y,i*size_x:(i+1)*size_x])

    # Given the contours, determine all the puzzle pieces
    def calc_pieces(self, margin=25, draw=False):
        """ Determine the pieces if they're seperated from eachother and there's black space in between"""
        # Initialise the used variables for finding the pieces
        self.pieces = []
        self.__new_pieces = []
        self.pieces_gr = []
        offset = int(margin/2)
        width, height = self.puzzle.shape[:2]
        
        # If we're drawing, make sure we don't draw on the original image
        if draw:
            mask = self.puzzle.copy()

        # If theres no contours calculated yet, make sure to do so
        if self.__contours is None:
            self.contours()

        # Create a bounding box with some margin around the contours
        for i in self.__contours:
            _, _, angle = cv2.minAreaRect(i)
            rect = cv2.boundingRect(i)
            x, y, w, h = rect
            y_start = y-offset
            x_start = x-offset
            y_end = y+h+offset
            x_end = x+w+offset

            # Make sure that the bounding boxes remain within the size of the image
            if y_start < 0:
                y_start = 0
            if x_start < 0:
                x_start = 0
            if y_end > height:
                y_end = height-1
            if x_end > width:
                x_end = width-1

            # Copy the puzzle piece from the image
            piece = self.puzzle[y_start:y_end,x_start:x_end].copy()

            # Based on the minAreaRect function we can use the angle of that to make sure the
            # images are all orientated the same way
            cols, rows = piece.shape[:2]
            rotM = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
            piece = cv2.warpAffine(piece, rotM, (cols, rows))

            # Store the pieces in a publicly available attribute of Puzzle
            self.pieces.append(piece)
            # Automaticly store a grayscale version of the puzzle piece
            self.pieces_gr.append(cv2.cvtColor(piece, cv2.COLOR_BGR2GRAY))

            # On top of a copy of the image, draw the bounding boxes
            if draw:
                cv2.rectangle(mask, (x_start, y_start), (x_end, y_end), (0, 255, 0), 1)

        # Once we have the pieces, there's still a lot of black margin around them (because of the space
        # needed for the rotation). We rerun the above code on each seperate piece so we get a better
        # puzzle piece to work with with less black borders in it.
        for piece, gray in zip(self.pieces, self.pieces_gr):
            self.__thresh = np.asarray((gray>0)*255,dtype= np.uint8)
            _, contours, hierarch = cv2.findContours(self.__thresh.copy(), cv2.RETR_EXTERNAL,
                                                      cv2.CHAIN_APPROX_NONE)
            for i in contours:
                _, _, angle = cv2.minAreaRect(i)
                rect = cv2.boundingRect(i)
                x, y, w, h = rect
                y_start = y+1
                x_start = x+1
                y_end = y+h-1
                x_end = x+w-1
                if y_start < 0:
                    y_start = 0
                if x_start < 0:
                    x_start = 0
                if y_end > height:
                    y_end = height-1
                if x_end > width:
                    x_end = width-1

                p = piece[y_start:y_end,x_start:x_end].copy()

                cols, rows = p.shape[:2]
                # Make sure that we don't have some weird anomalies that get threated as a puzzle piece
                if cols < 10 or rows < 10:
                    continue
                rotM = cv2.getRotationMatrix2D((cols/2, rows/2), int(angle), 1)
                p = cv2.warpAffine(p, rotM, (cols, rows))
                self.__new_pieces.append(p)

        if draw:
            cv2.imshow('test', mask)
            cv2.waitKey()
            cv2.destroyWindow('test')

        self.pieces = self.__new_pieces

        print('Found {} puzzle pieces.'.format(len(self.pieces)))

    def show_pieces(self, time=500, gray=False):
        """ Simply show all the puzzle pieces in the image """
        if self.pieces is None:
            self.calc_pieces()

        if gray:
            for p in self.pieces_gr:
                cv2.imshow('test', p)
                cv2.waitKey(time)
                cv2.destroyWindow('test')


        for p in self.pieces:
            cv2.imshow('test', p)
            cv2.waitKey(time)
            cv2.destroyWindow('test')

