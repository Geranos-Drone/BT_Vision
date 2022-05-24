import io
import numpy as np
import PIL
from PIL import Image
import torchvision
import requests
import torch
import openpifpaf

# Idea: make predictor preprocessing identical to train preprocessing!

#import constants
from openpifpaf_pole_plugin.constants import POLE_KEYPOINTS, POLE_SKELETON, POLE_SIGMAS, POLE_CATEGORIES, POLE_POSE, POLE_SCORE_WEIGHTS
from openpifpaf import encoder, headmeta, metric, transforms

import pdb


print('OpenPifPaf version', openpifpaf.__version__)
print('PyTorch version', torch.__version__)

cif = headmeta.Cif('cif', 'pole_detect',
                   keypoints=POLE_KEYPOINTS,
                   sigmas=POLE_SIGMAS,
                   pose=POLE_POSE,
                   draw_skeleton=POLE_SKELETON,
                   score_weights=POLE_SCORE_WEIGHTS)

caf = headmeta.Caf('caf', 'pole_detect',
                   keypoints=POLE_KEYPOINTS,
                   sigmas=POLE_SIGMAS,
                   pose=POLE_POSE,
                   skeleton=POLE_SKELETON)

head_metas = [cif, caf]

pil_im = Image.open("/home/tim/BT_Vision/convert_to_coco/test_dataset_coco/images/train/220511_1001.jpg")
img = np.asarray(pil_im)

predictor = openpifpaf.Predictor(checkpoint='outputs/mobilenetv2-220524-132344-pole_detect.pkl.epoch170', head_metas=head_metas) 
predictor.long_edge = 513 #does not change anything
predictions, gt_anns, image_meta = predictor.numpy_image(img)

if len(predictions) == 0:
    print("No Keypoints found!")

else:
    print(predictions[0].data)

input("Press Enter to stop...")