import openpifpaf

from .pole_detect_kp import PoleDetectKp

#TODO

def register():
    openpifpaf.DATAMODULES['pole_detect'] = pole_detect_kp.PoleDetectKp
