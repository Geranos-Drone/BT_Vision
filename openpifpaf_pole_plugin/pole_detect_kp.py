import argparse
import torch
import numpy as np
try:
    from pycocotools.coco import COCO
except ImportError:
    COCO = None

from openpifpaf.datasets import DataModule
from openpifpaf import encoder, headmeta, metric, transforms
from openpifpaf.datasets import collate_images_anns_meta, collate_images_targets_meta
from openpifpaf.plugins.coco import CocoDataset as CocoLoader

from .constants import POLE_KEYPOINTS, HFLIP, POLE_SKELETON, POLE_SIGMAS, POLE_CATEGORIES, POLE_POSE

class PoleDetecKp(DataModule):
    """
    DataModule for the Geranos Pole Dataset.
    """
    train_annotations = 'data/dataset_test.json'
    val_annotations = 'data/dataset_test.json'
    eval_annotations = val_annotations
    train_image_dir = 'images/'
    val_image_dir = 'images/'
    eval_image_dir = val_image_dir

    n_images = None
    square_edge = 513
    extended_scale = False
    orientation_invariant = 0.0
    blur = 0.0
    augmentation = True
    rescale_images = 1.0
    upsample_stride = 1
    min_kp_anns = 1
    b_min = 1  # 1 pixel

    eval_annotation_filter = True
    eval_long_edge = 0  # set to zero to deactivate rescaling
    eval_orientation_invariant = 0.0
    eval_extended_scale = False

    def __init__(self):
        super().__init__()
        # CIF (Composite Intensity Fields) to detect keypoints, and CAF (Composite Association Fields) to associate joints
        cif = headmeta.Cif('cif', 'pole_detect',
                           keypoints=self.POLE_KEYPOINTS,
                           sigmas=self.POLE_SIGMAS,
                           pose=self.POLE_POSE,
                           draw_skeleton=self.POLE_SKELETON,
                           score_weights=self.POLE_SCORE_WEIGHTS)

        caf = headmeta.Caf('caf', 'pole_detect',
                           keypoints=self.POLE_KEYPOINTS,
                           sigmas=self.POLE_SIGMAS,
                           pose=self.POLE_POSE,
                           skeleton=self.POLE_SKELETON)

        cif.upsample_stride = self.upsample_stride
        caf.upsample_stride = self.upsample_stride
        self.head_metas = [cif, caf]

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('data module Apollo')

        group.add_argument('--pole_detect-train-annotations',
                           default=cls.train_annotations)
        group.add_argument('--pole_detect-val-annotations',
                           default=cls.val_annotations)
        group.add_argument('--pole_detect-train-image-dir',
                           default=cls.train_image_dir)
        group.add_argument('--pole_detect-val-image-dir',
                           default=cls.val_image_dir)

        group.add_argument('--pole_detect-square-edge',
                           default=cls.square_edge, type=int,
                           help='square edge of input images')
        assert not cls.extended_scale
        group.add_argument('--pole_detect-extended-scale',
                           default=False, action='store_true',
                           help='augment with an extended scale range')
        group.add_argument('--pole_detect-orientation-invariant',
                           default=cls.orientation_invariant, type=float,
                           help='augment with random orientations')
        group.add_argument('--pole_detect-blur',
                           default=cls.blur, type=float,
                           help='augment with blur')
        assert cls.augmentation
        group.add_argument('--pole_detect-no-augmentation',
                           dest='apollo_augmentation',
                           default=True, action='store_false',
                           help='do not apply data augmentation')
        group.add_argument('--pole_detect-rescale-images',
                           default=cls.rescale_images, type=float,
                           help='overall rescale factor for images')
        group.add_argument('--pole_detect-upsample',
                           default=cls.upsample_stride, type=int,
                           help='head upsample stride')
        group.add_argument('--pole_detect-min-kp-anns',
                           default=cls.min_kp_anns, type=int,
                           help='filter images with fewer keypoint annotations')
        group.add_argument('--pole_detect-bmin',
                           default=cls.b_min, type=int,
                           help='b minimum in pixels')

        # evaluation
        assert cls.eval_annotation_filter
        group.add_argument('--pole_detect-no-eval-annotation-filter',
                           dest='apollo_eval_annotation_filter',
                           default=True, action='store_false')
        group.add_argument('--pole_detect-eval-long-edge', default=cls.eval_long_edge, type=int,
                           help='set to zero to deactivate rescaling')
        assert not cls.eval_extended_scale
        group.add_argument('--pole_detect-eval-extended-scale', default=False, action='store_true')
        group.add_argument('--pole_detect-eval-orientation-invariant',
                           default=cls.eval_orientation_invariant, type=float)

     @classmethod
    def configure(cls, args: argparse.Namespace):
        # extract global information
        cls.debug = args.debug
        cls.pin_memory = args.pin_memory

        # Apollo specific
        cls.train_annotations = args.pole_detect_train_annotations
        cls.val_annotations = args.pole_detect_val_annotations
        cls.eval_annotations = cls.val_annotations
        cls.train_image_dir = args.pole_detect_train_image_dir
        cls.val_image_dir = args.pole_detect_val_image_dir
        cls.eval_image_dir = cls.val_image_dir

        cls.square_edge = args.pole_detect_square_edge
        cls.extended_scale = args.pole_detect_extended_scale
        cls.orientation_invariant = args.pole_detect_orientation_invariant
        cls.blur = args.pole_detect_blur
        cls.augmentation = args.pole_detect_augmentation  # loaded by the dest name
        cls.rescale_images = args.pole_detect_rescale_images
        cls.upsample_stride = args.pole_detect_upsample
        cls.min_kp_anns = args.pole_detect_min_kp_anns
        cls.b_min = args.pole_detect_bmin

        # evaluation
        cls.eval_annotation_filter = args.pole_detect_eval_annotation_filter
        cls.eval_long_edge = args.pole_detect_eval_long_edge
        cls.eval_orientation_invariant = args.pole_detect_eval_orientation_invariant
        cls.eval_extended_scale = args.pole_detect_eval_extended_scale

    def _preprocess(self):

    def train_loader(self):

    def val_loader(self):

    @classmethod
    def common_eval_preprocess(cls):

    def _eval_preprocess(self):

    def eval_loader(self):

    def metrics(self):