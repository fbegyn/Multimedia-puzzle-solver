import cv2
import numpy as np
import itertools

class puzzleSolver:
    # puzzleSolver(puzzles [], type)
    def __init__(self, puzzles):
        """ Initialisation code for the puzzle solver """
        self.puzzle = puzzles
        self.__orb = cv2.ORB_create()
        self.__bf= cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # TODO: Put this in the puzzle class, doesn't really belong here
    def slice_image(self, dimv, dimh):
        """ Slice the image into it's seperate puzzle pieces and store them """
        size_x = int(len(self.puzzle.puzzle)/dimv)
        size_y = int(len(self.puzzle.puzzle[0])/dimh)
        self.puzzle.dimv = dimv
        self.puzzle.dimh = dimh
        self.puzzle.piece_v = size_x
        self.puzzle.piece_h = size_y
        self.puzzle.pieces = np.empty([dimv*dimh, size_x, size_y,3])
        for i in range(dimh):
            for j in range(dimv):
                self.puzzle.pieces[i*dimv+j] = self.puzzle.puzzle[j*size_x:(j+1)*size_x, i*size_y:(i+1)*size_y]
        
    def compare_rgb_pixels(pixel1, pixel2):
        """ Compares 2 pixel based on the RGB value. Comparriosn happens
            by taking the difference between the values and then calculating
            the pythogyrian distance between the results """
        b = abs(int(pixel1[0])-int(pixel2[0]))
        r = abs(int(pixel1[1])-int(pixel2[1]))
        g = abs(int(pixel1[2])-int(pixel2[2]))
        return int(np.sqrt(b**2 + r**2 + g**2))
        
    def compare_rgb_slices(this, slice1, slice2):
        """ Compare 2 image slices based on the compare_rgb_pixels funtion. The
            comparisson happens by calculating the average result from compare_
            rgb_pixles """
        weight = 0
        len1 = len(slice1)
        len2 = len(slice2)
        
        if(len1==len2):
            for i in range(len1):
                weight += puzzleSolver.compare_rgb_pixels(slice1[i], slice2[i])
        else:
            return 9999#256*3*max(len1, len2)
            #return 4095
        return int(weight/len1)
        
    def get_edges_nesw_clockwise(self, piece_number, step):
        """ Create slices from the image in a clockwise way """
        north = self.puzzle.pieces[piece_number][0,0::step]
        east = self.puzzle.pieces[piece_number][0::step,-1]
        south = self.puzzle.pieces[piece_number][-1,0::step][::-1]
        west = self.puzzle.pieces[piece_number][0::step,0][::-1]
        return (north, east, south, west)
    
    def get_edges_nesw_counterclockwise(self, piece_number, step):
        """ Create slices from the image in counterclockwise manner """
        north = self.puzzle.pieces[piece_number][0,0::step][::-1]
        east = self.puzzle.pieces[piece_number][0::step,-1][::-1]
        south = self.puzzle.pieces[piece_number][-1,0::step]
        west = self.puzzle.pieces[piece_number][0::step,0]
        return (north, east, south, west)
    
    def compare_two_indexed_pieces(self, index1, index2):
        """ Compare the 4 sides of 2 indexed puzzle pieces and return
            the matches based on compare_rgb_slices """
        step = 2
        slices1 = self.get_edges_nesw_clockwise(index1, step)
        slices2 = self.get_edges_nesw_counterclockwise(index2, step)
        # [[ n1n2, n1e2, n1s2, n1w2 ]       [ n1: [ n2, e2, s2, w2 ]
        #  [ e1n2, e1e2, e1s2, e1w2 ]         e1: [ n2, e2, s2, w2 ]
        #  [ s1n2, s1e2, s1s2, s1w2 ]         s1: [ n2, e2, s2, w2 ]
        #  [ w1n2, w1e2, w1s2, w1w2 ]]        w1: [ n2, e2, s2, w2 ] ]
        matches = np.empty([4,4], dtype=int)
        for i in (0,1,2,3):
            for j in (0,1,2,3):
                matches[i,j] = self.compare_rgb_slices(slices1[i], slices2[j])
        return matches

    def min_of_array(self, array):
        """ returns the minimun value of the array """
        min_index=0
        min_val=array[0]
        for i in (0,1,2,3):
            if(min_val>array[i]):
                min_index = i
                min_val = array[i]
        return (min_index, min_val)
    
    def get_mapper(self):

        pieces_amount = len(self.puzzle.pieces)
        piece_indexes = np.arange(pieces_amount)
        
        matches = np.empty([pieces_amount, pieces_amount,4,4], dtype=int)

        # generate all posible piece combinations excluding permutation
        combi = itertools.combinations(piece_indexes, r=2)
        # calculate de matching weight of each edge
        # for each combination of pieces
        for (a,b) in combi:
            # Determine the matching between the 2 pieces
            temp = self.compare_two_indexed_pieces(a, b)
            # Store the match result on place a, b in the matrix
            matches[a,b] = temp
            # Logically, if a matches b, b also matches a in the same way
            matches[b,a] = np.transpose(temp)
            
        # generate all posible piece combinations including permutation
        product =  itertools.product(piece_indexes, repeat=2)
        mapper = np.empty([pieces_amount, 4, 3])
        mapper[:,:,2] = 9999
        for (a,b) in product:
            #skip matching of piece with its self
            if(a != b):
                #for each edge: 0 = north, 1 = east, 2 = south, 3 = west
                for i in (0,1,2,3):
                    #find for piece 'a' with edge 'i' the best edge of piece 'b'
                    match = self.min_of_array(matches[a,b,i])
                    #if this piece is a better match 
                    if(match[1] < mapper[a,i,2]):
                        # then update the edge 'i' of piece 'a'
                        mapper[a,i] = [b,match[0], match[1]]
        return mapper

        
    def get_best_match_from_mapper(self, mapper, best_start):
        dimv = self.puzzle.dimv
        dimh = self.puzzle.dimh
        match = np.empty([dimv, dimh,2], dtype=int)
        #TODO find good startingpoint on left corner
        match[0,0] = [best_start, 0]

        #TODO add rotation
        for i in range(1,dimv):
            #get the parrent piece
            parrent =        match[i-1,0]
            #get best pice that matches with the adjesent edge of the parrent.
            #parrent[0] = piece number
            #parrent[1] = edge that is facing upward, example:
            # if parrent[1] == 3 then the the west side is faced upwards
            #to get best pice below the parrent piece rot+2 is needed
            best = mapper[parrent[0],(parrent[1]+2)%4]
            match[i,0] = [best[0] , best[1]]

        for i in range(dimv):
            for j in range(1,dimh):
                parrent = match[i,j-1]
                #parrent[1] is the side that faces upwards so rot+1 is needed
                #to find the best piece to its right
                best = mapper[parrent[0],(parrent[1]+1)%4]
                #the left edge of this piece is matched with the parrent
                #the upward facing edge is therby left edge+1
                match[i,j] = [best[0] , (best[1]+1)%4]

        return match
    
    def get_solution_from_best_match(self, best_match):

        dimv = self.puzzle.dimv
        dimh = self.puzzle.dimh
        piece_v = self.puzzle.piece_v
        piece_h = self.puzzle.piece_h
        
        solution = np.empty([piece_v * dimv, piece_h * dimh,3], dtype = np.uint8)
 
        for i in range(dimh):
            for j in range(dimv):
                #TODO add rotation logic
                #rot = best_match[j,i,1]
                #rot = 0 => no rotation
                #rot = 1 => +90 (counter-clockwise)
                #rot = 2 => 180
                #rot = 3 => -90 (clockwise)
                piece = self.puzzle.pieces[best_match[j,i,0]]
                rot   = best_match[j,i,1]
                rotated_piece = piece # = rotation_logic(piece, rot)
                solution[j*piece_v:(j+1)*piece_v, i*piece_h:(i+1)*piece_h] = rotated_piece 
        return solution
        
    
    
    
    
    
    
    
    
    
