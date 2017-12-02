# Multimedia-puzzle-solver
University project to make an image based puzzle solver

## Dependencies
* Python 3.6
* OpenCV 3.3.0
* Numpy 1.13.3

## Usage
The code needs to be executed from within the src directory (relative paths).
Simply run the main.py file with the correct puzzles passed to the function, and it will solve those puzzles.

## Structure
The program exists out of 2 classes, a Puzzle and a Puzzlesolver class.
The Puzzle class doe smost of the preproccessing needed to solve the puzzles. It stores the image and a grayscale version of it.
Afterwards it has the capability to fetch seperate puzzle piees from the scambled images with a relatively small error margin.
Although currently we haven't been able to make sure our algorithm works on the scambled parts because there;s still a relatively 
large black border surrounding the small pieces.

The Puzzlesolver class is what actuall solves the puzzles by using our algorithm. Multiple puzzles cna bepassed to the same PuzzleSolver.