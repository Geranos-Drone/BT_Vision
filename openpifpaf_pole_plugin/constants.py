import os

import numpy as np

POLE_KEYPOINTS = [
    'top_right',       # 1
    'top_left',        # 2
    'bottom_right',    # 3
    'bottom_left',     # 4
]

HFLIP = {
    'top_right': 'top_left',
    'top_left': 'top_right',
    'bottom_right': 'bottom_left',
    'bottom_left': 'bottom_right',
}

POLE_SKELETON = [
    (1,2), (2,4), (4,3), (3,1)
]

#Pole Sigmas onlydefault values atm!
POLE_SIGMAS = [
    0.1,
    0.1,
    0.1,
    0.1
]

POLE_CATEGORIES = ['pole']

#Pole Pose onlydefault values atm!
POLE_POSE = np.array([
        [0.0, 4.3, 2.0],  # 'top_right',        # 1
        [-0.4, 4.7, 2.0],  # 'top_left',        # 2
        [0.4, 4.7, 2.0],  # 'bottom_right',     # 3
        [-0.7, 5.0, 2.0],  # 'bottom_left',     # 4
    ])

#Score Weights copied from Animal KP
split, error = divmod(len(POLE_KEYPOINTS), 4)
POLE_SCORE_WEIGHTS = [5.0] * split + [3.0] * split + [1.0] * split + [0.5] * split + [0.1] * error


assert len(POLE_KEYPOINTS) == len(POLE_SIGMAS), "dimensions!"