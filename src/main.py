


import lib.puzzle as puzzle
import lib.puzzleSolver as puzzleSolver
import numpy as np
import cv2
import glob
from timeit import default_timer as timer
import matplotlib.pyplot as plt

def show(img, title='Puzzle', time=0):
    cv2.imshow(title,img)
    cv2.waitKey(time)
    cv2.destroyWindow(title)

puzzle_list= []
folder = "../puzzles/Tiles/tiles_rotated/"
for path in sorted(glob.glob(folder + '*.png')):
    #print(path)
    filename = path.split("\\")[1]
    dim = filename.split("_")[2].split("x")
    puzzle_list.append([puzzle.Puzzle(path), int(dim[0]), int(dim[1]), filename])


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
'''
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
show(np.hstack([aa, bb]), "bb")
show(np.hstack([aa, cc]), "cc")

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



'''
avg_time = 0
time = timer()
#for i in range(10):
for p in [puzzle_list[36]] :
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
'''


def compare_rgb_pixels2(pixel1, pixel2):
        b = abs(pixel1[0]-pixel2[0])
        r = abs(pixel1[1]-pixel2[1])
        g = abs(pixel1[2]-pixel2[2]) 
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

aaa = aa[0::,-1].astype(int)
bbb = bb[0::,0].astype(int)
ccc = cc[0::,0].astype(int)

def magnitude_spec(spec):
    magnitude_spectrum = np.log(np.abs(spec)+1)
    magnitude_spectrum += np.min(magnitude_spectrum) 
    magnitude_spectrum *= 255./np.max(magnitude_spectrum)
    return magnitude_spectrum

def fft_tf(image):
    fbeeld = np.fft.fft2(image) 
    fshift = np.fft.fftshift(fbeeld) 
    return fshift



afftb = np.fft.fft(aaa[:,0])
afftb = np.fft.fftshift(afftb)
afftb = np.log(np.abs(afftb)+1)
afftb -= np.min(afftb)
afftb *= 255./np.max(afftb)
print(afftb)

bfftb = np.fft.fft(bbb[:,0])
bfftb = np.fft.fftshift(bfftb)
bfftb = np.log(np.abs(bfftb)+1)
bfftb -= np.min(bfftb)
bfftb *= 255./np.max(bfftb)
print(bfftb)

cfftb = np.fft.fft(ccc[:,0])
cfftb = np.fft.fftshift(cfftb)
cfftb = np.log(np.abs(cfftb)+1)
cfftb -= np.min(cfftb)
cfftb *= 255./np.max(cfftb)
print(cfftb)


plt.plot(afftb)
plt.plot(bfftb)
plt.show()

plt.plot(afftb)
plt.plot(cfftb)
plt.show()

telab = np.sum(abs(afftb - bfftb))

print(abs(afftb - bfftb))
telac = np.sum(abs(afftb - cfftb))

print("res: ", "bbb: ",telab,"ccc: ", telac)






'''
sum1 = 0
sum2 = 0
for i in range(len(aaa)):
    temp1 = compare_rgb_pixels2(aaa[i],bbb[i])
    temp2 = compare_rgb_pixels2(aaa[i],ccc[i])
    #print("bb: ",temp1, "cc: ",temp2)
    sum1 += temp1
    sum2 += temp2

print("res: ", "bbb: ",sum1,"ccc: ", sum2)

'''





















