import os

import numpy as np

POLE_KEYPOINTS = [
    'tip',        #0
    'top_left',       #1
    'top_right',        #2
    'mid_left',        #3
    'mid_right',       #4
    'bottom_left',    #5
    'bottom_right',       #6
]

# At this pointnot in use -> ignored
HFLIP = {
    'top_right': 'top_left',
    'top_left': 'top_right',
    'mid_left': 'mid_right',
    'mid_right': 'mid_left',
    'bottom_right': 'bottom_left',
    'bottom_left': 'bottom_right',
}

POLE_SKELETON = [
    [0,1],[0,2],[1,2],[1,3],[2,4],[3,4],[3,5],[4,6],[5,6]
]

#Pole Sigmas onlydefault values atm!
POLE_SIGMAS = [
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1
]

POLE_CATEGORIES = ['pole']

#Pole Pose onlydefault values atm!
POLE_POSE = np.array([
        [0.0, 0.0, 1.3], # tip
        [-0.07, 0.0, 1.1], # top_l
        [0.0, 0.07, 1.1], # top_r
        [-0.07, 0.0, 0.55], # mid_l
        [0.0, 0.07, 0.55], # mid_r
        [-0.07, 0.0, 0.0], # bottom_l
        [0.0, 0.07, 0.0], # bottom_r
    ])

#Score Weights copied from Animal KP
split, error = divmod(len(POLE_KEYPOINTS), 4)
POLE_SCORE_WEIGHTS = [5.0] * split + [3.0] * split + [1.0] * split + [0.5] * split + [0.1] * error


assert len(POLE_KEYPOINTS) == len(POLE_SIGMAS), "dimensions!"
