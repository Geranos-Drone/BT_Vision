import os

import numpy as np

POLE_KEYPOINTS = [
    'star',     # 1
    'rhomb',    # 2
    'torp',     # 3
    'tear',     # 4
    'arrow',    # 5
    'triangle', # 6
    'ninja',    # 7
    'bolt'      # 8
]

# At this pointnot in use -> ignored
HFLIP = {
    'top_right': 'top_left',
    'top_left': 'top_right',
    'bottom_right': 'bottom_left',
    'bottom_left': 'bottom_right',
}

POLE_SKELETON = [
    [1, 2], [1, 4], [1, 5], [2, 3], 
    [2, 6], [3, 4], [3, 7], [4, 8], 
    [5, 8], [5, 6], [6, 7], [7, 8]
]

#Pole Sigmas onlydefault values atm!
POLE_SIGMAS = [
    0.1,
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
        [0.0, 1.0, 0.0],    # 'star',       # 1
        [1.0, 0.0, 0.0],    # 'rhomb',      # 2
        [-1.0, 0.0, 0.0],   # 'torp',       # 3
        [0.0, -1.0, 0.0],   # 'tear',       # 4
        [0.0, 1.0, 2.0],    # 'arrow',      # 5
        [1.0, 0.0, 2.0],    # 'triangle',   # 6
        [-1.0, 0.0, 2.0],   # 'ninja',      # 7
        [0.0, -1.0, 2.0],   # 'bolt',       # 8
    ])

#Score Weights copied from Animal KP
split, error = divmod(len(POLE_KEYPOINTS), 4)
POLE_SCORE_WEIGHTS = [5.0] * split + [3.0] * split + [1.0] * split + [0.5] * split + [0.1] * error


assert len(POLE_KEYPOINTS) == len(POLE_SIGMAS), "dimensions!"