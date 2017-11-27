


import lib.puzzle as puzzle
import lib.puzzleSolver as puzzleSolver
import numpy as np
import cv2
import glob
from timeit import default_timer as timer

def show(img, title='Puzzle', time=0):
    cv2.imshow(title,img)
    cv2.waitKey(time)
    cv2.destroyWindow(title)

puzzle_list= []
folder = "../puzzles/Tiles/tiles_rotated/"
for path in sorted(glob.glob(folder + '*.png')):
    filename = path.split("/")[4]
    dim = filename.split("_")[2].split("x")
    puzzle_list.append([puzzle.Puzzle(path), int(dim[0]), int(dim[1]), filename])

'''
def compare_rgb_pixels(pixel1, pixel2):
        
        LL = pixel1[0]
        LR = pixel1[1]
        RL = pixel2[0]
        RR = pixel2[1]
        #print(LL,"\t", LR,"\t", RL,"\t", RR)
        distR = LR + (LR - LL)
        distL = RL + (RL - RR)
        tresh = 70
        b = (abs(RL[0]-distR[0]) + abs(LR[0]-distL[0]) + abs(RL[0]-LR[0])) < tresh
        r = (abs(RL[1]-distR[1]) + abs(LR[1]-distL[1]) + abs(RL[1]-LR[1])) < tresh
        g = (abs(RL[2]-distR[2]) + abs(LR[2]-distL[2]) + abs(RL[2]-LR[2])) < tresh

        #b = abs(int(pixel1[1][0])-int(pixel2[0][0])) + 0.5*abs(int(pixel1[0][0])-int(pixel2[1][0]))
        #r = abs(int(pixel1[1][1])-int(pixel2[0][1])) + 0.5*abs(int(pixel1[0][1])-int(pixel2[1][1]))
        #g = abs(int(pixel1[1][2])-int(pixel2[0][2])) + 0.5*abs(int(pixel1[0][2])-int(pixel2[1][2])
        #return int(np.sqrt(b**2 +r**2 +g**2))
        return int(b+r+g)
        #return int(max(b,r,g))

a = 1
b = 8
c= 13

solver = puzzleSolver.puzzleSolver(puzzle_list[36][0])
solver.slice_image(5,5)
aa = solver.puzzle.pieces[a]
bb = solver.puzzle.pieces[b]
cc = solver.puzzle.pieces[c]

aa = np.rot90(aa,0)
bb = np.rot90(bb,1)
cc = np.rot90(cc,0)
#show(np.hstack([aa, bb]), "bb")
#show(np.hstack([aa, cc]), "cc")

alen = len(aa)
blen = len(bb)
clen = len(cc)

aaa = aa[0::,-2:].astype(int)
bbb = bb[0::,0:2].astype(int)
ccc = cc[0::,0:2].astype(int)
#print(aaa)

sum1 = 0
sum2 = 0
for i in range(len(aaa)):
    temp1 = compare_rgb_pixels(aaa[i],bbb[i])
    #print("bb: ",temp1, "cc: ",temp2)
    sum1 += temp1

print("\n\n\ccc\n")
    
for i in range(len(ccc)):
    temp2 = compare_rgb_pixels(aaa[i],ccc[i])
    #print("bb: ",temp1, "cc: ",temp2)
    sum2 += temp2
    

print("res: ", "bbb: ",sum1,"ccc: ", sum2)



'''
#[puzzle_list[36]]

avg_time = 0
time = timer()
#for i in range(10):
for p in puzzle_list:
        solver = puzzleSolver.puzzleSolver(p[0])
        solver.slice_image(p[1],p[2])
        mapper  = solver.get_mapper()
        match = solver.get_best_match_from_mapper(mapper)
        solution = solver.get_solution_from_best_match(match)
        #comment out show command for accurat timing info.
        show(solution, title=p[3])
timing = timer() - time
print("Total execution time:")
print(np.ceil(timing*1000)/1000, "sec")























