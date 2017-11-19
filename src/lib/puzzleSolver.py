import cv2
import numpy as np
import itertools

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
        self.puzzle.dimv = dimv
        self.puzzle.dimh = dimh
        self.puzzle.piece_v = size_x
        self.puzzle.piece_h = size_y
        slices = [np.empty([size_x, size_y,3]) for _ in range(dimv*dimh)]
        for i in range(dimh):
            for j in range(dimv):
                slices[i*dimv+j] = self.puzzle.puzzle[j*size_x:(j+1)*size_x, i*size_y:(i+1)*size_y]
        self.puzzle.pieces = slices
    
    def compare_rgb_pixels(pixel1, pixel2):
        b = abs(int(pixel1[0])-int(pixel2[0]))
        r = abs(int(pixel1[1])-int(pixel2[1]))
        g = abs(int(pixel1[2])-int(pixel2[2]))
        return np.sqrt(b**2 + r**2 + g**2)
        
    def compare_rgb_slices(this, slice1, slice2):
        weight = 0
        len1 = len(slice1)
        len2 = len(slice2)
        
        if(len1==len2):
            for i in range(0,len1,2):
                weight += puzzleSolver.compare_rgb_pixels(slice1[i], slice2[i])
        else:
            return 999#256*3*max(len1, len2)
            #return 4095
        return weight/len1
        
    def get_edges_nesw_clockwise(self, piece_number):
        north = self.puzzle.pieces[piece_number][0,0:]
        east = self.puzzle.pieces[piece_number][0:,-1]
        south = self.puzzle.pieces[piece_number][-1,0:][::-1]
        west = self.puzzle.pieces[piece_number][0:,0][::-1]
        return [north, east, south, west]
    
    def get_edges_nesw_counterclockwise(self, piece_number):
        north = self.puzzle.pieces[piece_number][0,0:][::-1]
        east = self.puzzle.pieces[piece_number][0:,-1][::-1]
        south = self.puzzle.pieces[piece_number][-1,0:]
        west = self.puzzle.pieces[piece_number][0:,0]
        return [north, east, south, west]
    
        
    def get_pieces_combo_list(self):
        piece_indexes = np.arange(len(self.puzzle.pieces))
        return np.array(list(itertools.product(piece_indexes, repeat=2)))
        
    def compare_two_indexed_pieces(self, index1, index2):
        slices1 = self.get_edges_nesw_clockwise(index1)
        slices2 = self.get_edges_nesw_counterclockwise(index2)
        # n1n2, n1e2, n1s2, n1w2,
        # e1n2, e1e2, e1s2, e1w2,
        # s1n2, s1e2, s1s2, s1w2,
        # w1n2, w1e2, w1s2, w1w2
        matches = np.empty([4,4], dtype=np.uint64)
        for i in range(4):
            for j in range(4):
                matches[i,j] = self.compare_rgb_slices(slices1[i], slices2[j])
        return matches
    
    #unused
    def get_rotation(self, matches_min_index):
        rot2 = int(matches_min_index%4)
        rot1 = int((matches_min_index-rot2)/4)
        
        return [rot1, rot2]
    
    def min_of_array(self, array):
        min_index=0
        min_val=array[0]
        for i in range(len(array)):
            if(min_val>array[i]):
                min_index = i
                min_val = array[i]
        return [min_index, min_val]
    
    def get_mapper(self):
        combi = self.get_pieces_combo_list()
        combi_amount = len(combi)
        pieces_amount = len(self.puzzle.pieces)
        matches = np.empty([combi_amount,4,4], dtype=np.uint64)
        
        for i in range(combi_amount):
            matches[i] = self.compare_two_indexed_pieces(combi[i,0], combi[i,1])
        
        mapper = np.empty([pieces_amount, 4, 3])
        mapper[:,:,2] = 999
        match_north = [999,999]
        match_east = [999,999]
        match_south = [999,999]
        match_west = [999,999]
        for i in range(combi_amount):
            
            if(combi[i,0] != combi[i,1]):
                temp = self.min_of_array(matches[i,0])
                if(temp[1] < mapper[combi[i,0],0,2]):
                    match_north = temp
                    mapper[combi[i,0],0] = [combi[i,1],match_north[0], match_north[1]]
                    #mapper[combi[i,1],2] = [combi[i,0],0, match_north[1]]
                
                temp  = self.min_of_array(matches[i,1])
                if(temp[1]< mapper[combi[i,0],1,2]):
                    match_east = temp
                    mapper[combi[i,0],1] = [combi[i,1],match_east[0],match_east[1]]
                    #mapper[combi[i,1],3] = [combi[i,0],1,match_east[1]]
                
                temp = self.min_of_array(matches[i,2])
                if(temp[1]< mapper[combi[i,0],2,2]):
                    match_south = temp
                    mapper[combi[i,0],2] = [combi[i,1],match_south[0],match_south[1]]
                    #mapper[combi[i,1],0] = [combi[i,0],2,match_south[1]]
                
                temp = self.min_of_array(matches[i,3])
                if(temp[1]< mapper[combi[i,0],3,2]):
                    match_west  = temp
                    mapper[combi[i,0],3] = [combi[i,1],match_west[0],match_west[1]]
                    #mapper[combi[i,1],1] = [combi[i,0],3,match_west[1]]
                    #print(mapper[combi[i,0],3], mapper[combi[i,1],1])
        return mapper

        
    def get_best_match_from_mapper(self, mapper, best_start):
        #TODO get rid of extra border
        dimv = self.puzzle.dimv+1
        dimh = self.puzzle.dimh+1
        
        match = np.zeros([dimv, dimh], dtype=np.int8)
        match[0,:]=-1
        match[:,0]=-1
        
        #TODO find good startingpoint on left corner
        index_top_left = 0
        avg = mapper[0,0,2]*mapper[0,3,2] - mapper[0,1,2]*mapper[0,2,2]
        for i in range(len(mapper)):
            temp = mapper[i,0,2]*mapper[i,3,2]
            if(temp>avg):
                avg=temp
                index_top_left=i
        match[1,1] = best_start
        #TODO add rotation
        for i in range(2,dimv):
            match[i,1]=   mapper[match[i-1,1],2,0] 
        for i in range(1,dimv):
            for j in range(2,dimh):
                match[i,j]=   mapper[match[i,j-1],1,0]
        return match[1:, 1:]
    
    def get_solution_from_best_match(self, best_match):

        dimv = self.puzzle.dimv
        dimh = self.puzzle.dimh
        piece_v = self.puzzle.piece_v
        piece_h = self.puzzle.piece_h
        
        solution = np.zeros([piece_v * dimv, piece_h * dimh,3], dtype = np.uint8)
 
        for i in range(dimh):
            for j in range(dimv):
                solution[j*piece_v:(j+1)*piece_v, i*piece_h:(i+1)*piece_h] = self.puzzle.pieces[best_match[j,i]]
        return solution
        
    
    
    
    
    
    
    
    
    
