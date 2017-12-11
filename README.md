# Multimedia-puzzle-solver
University project to make an image based puzzle solver

## Dependencies
* Anaconda 3 spyder
* Python 3.6
* OpenCV 3.3.0
* Numpy 1.13.3
* Scipy 0.18.1

## Usage

The current main is written so that the current working directory is
set to src. So please go into src first before running `python main.py`.

location spyder: Anaconda3\Scripts\spyder.exe
File to execure with spyder: src\main.py
Before executing src/main.py, open the file

Sadly, UNIX and Windows have a different way of managing files.
UNIX uses /'s to descent in paths, and Windows uses \'s to decent into
directories.
If you are Windows user: 	comment-out line 21 and uncomment line 22
If you are Linux user: 		comment-out line 22 and uncomment line 21

## findContours, Suzukis algorithm in disguise

Suzukis algorithm classificieerd eerst de pizel gebasseerd op zijn lokaal gebied. Hierna begint het algoritme met het volgen van pixel s gebasserd op de classificatie van de vorige pixel.
