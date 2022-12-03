# OpenPifPaf for Pole Pose Esimation

### Training Command:

python3 -m openpifpaf.train --checkpoint=mobilenetv2 --dataset=pole_detect --epochs=170

### Predcit Command over CLI

python3 -m openpifpaf.predict <path_to_image> --checkpoint=<path_to_checkpoint> --long-edge=513


python3 -m openpifpaf.predict /home/tim/BT_Vision/convert_to_coco/test_dataset_coco_2/images/train/220511_1001.jpg --checkpoint=outputs/mobilenetv2-220525-151338-pole_detect.pkl.epoch300 --long-edge=513 --decoder=cifcaf

### Look into Log Files

python3 -m openpifpaf.logs <path_to_log_file> --show

python3 -m openpifpaf.logs /home/tim/BT_Vision/outputs/mobilenetv2-220523-163713-pole_detect.pkl.log --show

# Install Environment for Openpifpaf

1) Install PyTorch (pip cmd from website)
2) pip install openpifpaf
3) pip install pycocotools
4) pip install scipy
