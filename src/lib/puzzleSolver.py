import cv2
import numpy as np
import itertools
from timeit import default_timer as timer

class puzzleSolver:
    # puzzleSolver(puzzles [], type)
    def __init__(self, puzzles):
        """ Initialisation code for the puzzle solver """
        self.puzzle = puzzles
        #self.__orb = cv2.ORB_create()
        #self.__bf= cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # TODO: Put this in the puzzle class, doesn't really belong here
    def slice_image(self, dimv, dimh):
        """ Slice the image into it's seperate puzzle pieces and store them """
        size_x = int(len(self.puzzle.puzzle)/dimv)
        size_y = int(len(self.puzzle.puzzle[0])/dimh)
        self.puzzle.dimv = dimv
        self.puzzle.dimh = dimh
        self.puzzle.piece_v = size_x
        self.puzzle.piece_h = size_y
        self.puzzle.pieces = np.empty([dimv*dimh, size_x, size_y,3], dtype=int)
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
        return int(max(b,r,g))
        
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
            return 99999999
        return int(weight)
        
    def get_edges_nesw_clockwise(self, piece_number, step):
        """ Create slices from the image in a clockwise way """
        north = self.puzzle.pieces[piece_number][0,0::step].astype(int)
        east = self.puzzle.pieces[piece_number][0::step,-1].astype(int)
        south = self.puzzle.pieces[piece_number][-1,0::step][::-1].astype(int)
        west = self.puzzle.pieces[piece_number][0::step,0][::-1].astype(int)
        return (north, east, south, west)
    
    def get_edges_nesw_counterclockwise(self, piece_number, step):
        """ Create slices from the image in counterclockwise manner """
        north = self.puzzle.pieces[piece_number][0,0::step][::-1].astype(int)
        east = self.puzzle.pieces[piece_number][0::step,-1][::-1].astype(int)
        south = self.puzzle.pieces[piece_number][-1,0::step].astype(int)
        west = self.puzzle.pieces[piece_number][0::step,0].astype(int)
        return (north, east, south, west)
    
    def compare_two_indexed_pieces(self, index1, index2):
        """ Compare the 4 sides of 2 indexed puzzle pieces and return
            the matches based on compare_rgb_slices """
        step = 12
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
        mapper = np.empty([pieces_amount, 4, 3], dtype=int)
        mapper[:,:,2] = 99999999
        for (a,b) in product:
            #skip matching of piece with its self
            if(a != b):
                #for each edge: 0 = north, 1 = east, 2 = south, 3 = west
                for i in (0,1,2,3):
                    #find for piece 'a' with edge 'i' the best edge of piece 'b'
                    match = self.min_of_array(matches[a,b,i])
                    #if this piece is a better matchthen the previus
                    if(match[1] < mapper[a,i,2]):
                        # then update the mapping of edge 'i' of piece 'a'
                        mapper[a,i] = [b,match[0], match[1]]
        return mapper
        
    def get_best_match_from_mapper(self, mapper):
        dimv = self.puzzle.dimv
        dimh = self.puzzle.dimh
        match = np.empty([dimv, dimh,2], dtype=int)
        piece_indexes = np.arange(len(self.puzzle.pieces))
        all_seeds = itertools.product(piece_indexes,(0,1,2,3))
        best_match = np.empty_like(match)
        best_weight=99999999
        for (a,b) in all_seeds:
            weight = 0
            match[0,0] =[a,b]
            # keep track what puzzle pieces have been used already.
            pieces_mask = np.ones(len(self.puzzle.pieces))
            pieces_mask[a]=0
            #fill up upper horizontal row
            for j in range(1,dimh):
                parrent_left = match[0,j-1]
                #parrent_left[1] is a parrent's edge that faces upward.
                #(parrent_left[1]+1)%4 is parrent's right edge.
                best_piece_left = mapper[parrent_left[0],(parrent_left[1]+1)%4]
                #best_piece_left[1] is this piece's left edge
                #(best_piece_left[1]+1)%4 is this piece's edge that faces upward
                match[0,j] = [best_piece_left[0] , (best_piece_left[1]+1)%4]
                #best_piece_left[2] is the weight of the current match
                weight = weight + best_piece_left[2]
                #best_piece_left[0] is this piece's number
                if(pieces_mask[best_piece_left[0]]):
                    pieces_mask[best_piece_left[0]]=0
                else:
                    #if piece was already used then this match is invalid
                    weight = 99999999
                    #break out of this loop
                    break
                if(weight > best_weight):
                    #if after this match the wieght has exeeded a previos better
                    #match then there is no point anymore in continuing
                    break
            if(weight > best_weight):
                #continue wil skip the rest of the outer loop and continue to
                #the next iteration (next a and b)
                continue
            #fill up leftmost culom 
            for i in range(1,dimv):
                parrent_up = match[i-1,0]
                #when going dawnward then the bottom edge is the top edge+2 with
                #top edge beeing parrent_up[1]
                best_piece_up = mapper[parrent_up[0],(parrent_up[1]+2)%4]
                match[i,0] = best_piece_up[0:2]
                weight = weight + best_piece_up[2]
                if(pieces_mask[best_piece_up[0]]):
                    pieces_mask[best_piece_up[0]]=0
                else:
                    weight = 99999999

                if(weight > best_weight):
                    break
            if(weight > best_weight):
                continue
            #fill up the rest of the match by looking at bothe the top and left
            #neighbour
            for i in range(1,dimv):
                for j in range(1,dimh):
                    parrent_up = match[i-1,j]
                    map_up = mapper[parrent_up[0],(parrent_up[1]+2)%4]
                    best_piece_up = map_up[0:2]
                    
                    parent_left = match[i,j-1]
                    map_left = mapper[parent_left[0],(parent_left[1]+1)%4]
                    best_piece_left = [map_left[0] , (map_left[1]+1)%4][0:2]
                   
                    weight_up = map_up[2]
                    weight_left = map_left[2]
                    
                    ### STAY AWAY FROM THIS PART ###                    
                    if(best_piece_up[0] != best_piece_left[0]):
                        weight += 99999 #Do not alter anything here
                        #I don't know how this works but it does what is needed
                    ### STAY AWAY FROM THIS PART ###
                    
                    #pick one of the two( either one works)
                    match[i,j] = best_piece_up
                    
                    if(pieces_mask[map_up[0]]):
                        pieces_mask[map_up[0]]=0
                    else:
                        weight = 99999999
                    # add up both weights
                    weight = weight + weight_left + weight_up
                if(weight > best_weight):
                    break

            #if this match is better then update it
            if(weight < best_weight):
                best_weight = weight
                best_match = np.copy(match)
        return best_match
    
    def get_solution_from_best_match(self, best_match):
        dimv = self.puzzle.dimv
        dimh = self.puzzle.dimh
        piece_v = self.puzzle.piece_v
        piece_h = self.puzzle.piece_h
        #rotate the best_match in order to minimise image rotations
        #find the most frequent occuring top edge
        destribution = np.zeros(4);
        for rot in (best_match[:,:,1].flatten()):
            destribution[rot] +=1
        #print(best_match[:,:,1].flatten(),destribution )
        most_rot = np.argmax(destribution)
        #print(most_rot)
        #rotate the match matrix and update the top edge so that the north edge
        #becomes the most frequnet top edge. When a north edge is on top 
        #no rotation is needed. This also somhow fixes the issue of guessing
        #the verticval and horizontal dimentions of a rectangular image.
        adjusted_match = np.rot90(best_match,-most_rot)
        adjusted_match[:,:,1] = (adjusted_match[:,:,1]-most_rot)%4
        #TODO put solution into self.puzzle.solution
        solution = np.empty([piece_v * dimv, piece_h * dimh,3], dtype = np.uint8)
        for i in range(dimh):
            for j in range(dimv):
                #rot = 0 => no rotation
                #rot = 1 => +90 (counter-clockwise)
                #rot = 2 => 180
                #rot = 3 => -90 (clockwise)
                piece = self.puzzle.pieces[adjusted_match[j,i,0]]
                rot   = adjusted_match[j,i,1]
                if((rot%2) == 0):
                    solution[j*piece_v:(j+1)*piece_v, i*piece_h:(i+1)*piece_h] = np.rot90(piece,rot)
                else:
                    solution[j*piece_h:(j+1)*piece_h, i*piece_v:(i+1)*piece_v] = np.rot90(piece,rot)
        return solution
        
    
    
    
    
    
    
    
    
    
