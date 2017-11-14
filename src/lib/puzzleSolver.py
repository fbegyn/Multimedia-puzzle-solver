import cv2
import numpy as np

class puzzleSolver:
    # puzzleSolver(puzzles [], type)
    def __init__(self, puzzles):
        self.puzzle = puzzles
        self.__orb = cv2.ORB_create()
        self.__bf= cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def edge_match(self, edge1, edge2):
        weight = 0
        max_it = np.min(len(edge1), len(edge1))
        for i in range(max_it):
            weight += abs(edge1[i]- edge2[i])
        return weight
    
    def slice_image(self, dim):
        size_x = int(len(self.puzzle.puzzle[0])/dim)
        size_y = int(len(self.puzzle.puzzle)/dim)
        slices = []        
        for i in range(dim):
            for j in range(dim):
                slices.append(self.puzzle.puzzle[j*size_y:(j+1)*size_y, i*size_x:(i+1)*size_x])                
        self.puzzle.pieces = slices
            
        

