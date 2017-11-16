import cv2
import numpy as np

class puzzleSolver:
    # puzzleSolver(puzzles [], type)
    def __init__(self, puzzles):
        self.puzzle = puzzles
        self.__orb = cv2.ORB_create()
        self.__bf= cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def edge_match_grey(self, edge1, edge2):
        weight = 0
        max_it = np.min(len(edge1), len(edge1))
        for i in range(max_it):
            weight += abs(edge1[i]- edge2[i])
        return weight
    
    def slice_image(self, dimv, dimh):
        size_x = int(len(self.puzzle.puzzle)/dimv)
        size_y = int(len(self.puzzle.puzzle[0])/dimh)
        
        slices = [np.empty([size_x, size_y,3]) for _ in range(dimv*dimh)]
        for i in range(dimh):
            for j in range(dimv):
                slices[i*dimv+j] = self.puzzle.puzzle[j*size_x:(j+1)*size_x, i*size_y:(i+1)*size_y]
        self.puzzle.pieces = slices
    
    def compare_rgb_pixels(pixel1, pixel2):
        b = abs(int(pixel1[0])-int(pixel2[0]))
        r = abs(int(pixel1[1])-int(pixel2[1]))
        g = abs(int(pixel1[2])-int(pixel2[2]))
        return b+r+g
        
    def compare_rgb_slices(slice1, slice2):
        weight = 0
        if(len(slice1)==len(slice2)):
            for i in range(len(slice1)):
                weight = puzzleSolver.compare_rgb_pixels(slice1[i], slice2[i])
        else:
            return 256*3*max(len(slices1), len(slices2))
        return weight
        
    def get_edges_nesw(self, piece_number):
        north = self.puzzle.pieces[piece_number][0,0:]
        east = self.puzzle.pieces[piece_number][0:,-1]
        south = self.puzzle.pieces[piece_number][-1,0:]
        west = self.puzzle.pieces[piece_number][0:,0]
        return north, east, south, west
        
        
        
        
