import cv2
import numpy as np

class puzzleSolver:
    puzzleType = 'jigsaw'
    puzzles = []

    # puzzleSolver(puzzles [], type)
    def __init__(self, inp, puzzle_type='jigsaw'):
        self.puzzleType = puzzle_type
        for i in inp:
            self.puzzle.append(inp)
