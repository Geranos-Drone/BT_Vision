import io
import numpy as np
import PIL
from PIL import Image
import torchvision
import requests
import torch
import openpifpaf
import json

# Idea: make predictor preprocessing identical to train preprocessing!

#import constants
from openpifpaf_pole_plugin.constants import POLE_KEYPOINTS, POLE_SKELETON, POLE_SIGMAS, POLE_CATEGORIES, POLE_POSE, POLE_SCORE_WEIGHTS
from openpifpaf import encoder, headmeta, metric, transforms

import pdb


print('OpenPifPaf version', openpifpaf.__version__)
print('PyTorch version', torch.__version__)

# cif = headmeta.Cif('cif', 'pole_detect',
#                    keypoints=POLE_KEYPOINTS,
#                    sigmas=POLE_SIGMAS,
#                    pose=POLE_POSE,
#                    draw_skeleton=POLE_SKELETON,
#                    score_weights=POLE_SCORE_WEIGHTS)

# caf = headmeta.Caf('caf', 'pole_detect',
#                    keypoints=POLE_KEYPOINTS,
#                    sigmas=POLE_SIGMAS,
#                    pose=POLE_POSE,
#                    skeleton=POLE_SKELETON)

# head_metas = [cif, caf]
path_img = "/home/" + USERNAME + "/BT_Vision/convert_to_coco/vicon_dataset_coco/images/train/220602_1033.jpg"
path_network = "/home/" + USERNAME + "/BT_Vision/outputs/mobilenetv2-220622-202035-pole_detect.pkl.epoch900"
pil_im = Image.open(path_img)
img = np.asarray(pil_im)

#import ipdb; ipdb.set_trace()

predictor = openpifpaf.Predictor(checkpoint=path_network)
#predictor.long_edge = 513 #does not change anything
predictions, gt_anns, image_meta = predictor.numpy_image(img)
if len(predictions) == 0:
    print("No Keypoints found!")


else:
    print("found something")
    print(predictions[0].data)
    annotation_painter = openpifpaf.show.AnnotationPainter()
    with openpifpaf.show.image_canvas(img) as ax:
    	annotation_painter.annotations(ax, predictions)

input("Press Enter to stop...")
