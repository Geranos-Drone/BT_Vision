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

from .constants import POLE_KEYPOINTS, POLE_SKELETON, POLE_SIGMAS, POLE_CATEGORIES, POLE_POSE, POLE_SCORE_WEIGHTS

class PoleDetectKp(DataModule):
    """
    DataModule for the Geranos Pole Dataset.
    """
    train_annotations = '/home/tim/BT_Vision/convert_to_coco/test_dataset_coco/annotations/pole_keypoints_8_train.json'
    val_annotations = '/home/tim/BT_Vision/convert_to_coco/test_dataset_coco/annotations/pole_keypoints_8_val.json'
    eval_annotations = val_annotations
    train_image_dir = '/home/tim/BT_Vision/convert_to_coco/test_dataset_coco/images/train/'
    val_image_dir = '/home/tim/BT_Vision/convert_to_coco/test_dataset_coco/images/val/'
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

    batch_size = 8
    loader_workers = 0

    def __init__(self):
        super().__init__()
        # CIF (Composite Intensity Fields) to detect keypoints, and CAF (Composite Association Fields) to associate joints
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

        cif.upsample_stride = self.upsample_stride
        caf.upsample_stride = self.upsample_stride
        self.head_metas = [cif, caf]

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('data module to detect poles')

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
                           dest='pole_detect_augmentation',
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
                           dest='pole_detect_eval_annotation_filter',
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
        encoders = (encoder.Cif(self.head_metas[0], bmin=self.b_min),
                    encoder.Caf(self.head_metas[1], bmin=self.b_min))

        return transforms.Compose([
            transforms.NormalizeAnnotations(),
            transforms.Crop(self.square_edge, use_area_of_interest=True),
            transforms.TRAIN_TRANSFORM, #converts PIL.Image to np.array
            transforms.Encoders(encoders),
        ])

    def train_loader(self):
        train_data = CocoLoader(
            image_dir=self.train_image_dir,
            ann_file = self.train_annotations,
            preprocess=self._preprocess(),
            annotation_filter=True,
            min_kp_anns=self.min_kp_anns,
            category_ids=[420],
        )
        print("train_loader was called")
        print("batch size = ", self.batch_size)

        return torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=not self.debug,
            pin_memory=self.pin_memory, num_workers=self.loader_workers,
            drop_last=False, collate_fn = collate_images_targets_meta) #drop_last = True eigt.

    def val_loader(self):
        val_data = CocoLoader(
            image_dir=self.val_image_dir,
            ann_file=self.val_annotations,
            preprocess=self._preprocess(),
            annotation_filter=True,
            min_kp_anns=self.min_kp_anns,
            category_ids=[420],
        )
        return torch.utils.data.DataLoader(
            val_data, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=False, 
            collate_fn=collate_images_targets_meta) #drop_last = True eigt.

    @classmethod
    def common_eval_preprocess(cls):
        return [transforms.NormalizeAnnotations(),]

    def _eval_preprocess(self):
        return transforms.Compose([
            *self.common_eval_preprocess(),
            transforms.ToAnnotations([
                transforms.ToKpAnnotations(
                    POLE_CATEGORIES,
                    keypoints_by_category={420: self.head_metas[0].keypoints},
                    skeleton_by_category={420: self.head_metas[1].skeleton},
                ),
                transforms.ToCrowdAnnotations(POLE_CATEGORIES),
            ]),
            transforms.EVAL_TRANSFORM,
        ])

    def eval_loader(self):
        eval_data = CocoLoader(
            image_dir=self.eval_image_dir,
            ann_file=self.eval_annotations,
            preprocess=self._eval_preprocess(),
            annotation_filter=self.eval_annotation_filter,
            min_kp_anns=self.min_kp_anns if self.eval_annotation_filter else 0,
            category_ids=[420] if self.eval_annotation_filter else [420],
        )
        return torch.utils.data.DataLoader(
            eval_data, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=False,
            collate_fn=collate_images_anns_meta)

    def metrics(self):
        return [metric.Coco(COCO(self.eval_annotations),
                max_per_image=20,
                category_ids=[420],
                iou_type='keypoints',
                keypoint_oks_sigmas=POLE_SIGMAS,)]