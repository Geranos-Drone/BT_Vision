# OpenPifPaf

### Training Command:

python3 -m openpifpaf.train --checkpoint=mobilenetv2 --dataset=pole_detect --epochs=20

### Predcit Command over CLI

python3 -m openpifpaf.predict <path_to_image> --checkpoint=<path_to_checkpoint> --long_edge=513


python3 -m openpifpaf.predict /home/tim/BT_Vision/convert_to_coco/test_dataset_coco/images/train/220511_1001.jpg --checkpoint=mobilenetv2-220523-111436-pole_detect.pkl.epoch473 --long_edge=513

### Look into Log Files

python3 -m openpifpaf.logs <path_to_log_file> --show

python3 -m openpifpaf.logs /home/tim/BT_Vision/outputs/mobilenetv2-220523-163713-pole_detect.pkl.log --show
