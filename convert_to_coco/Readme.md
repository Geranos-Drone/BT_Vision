## Instructions for LSjson to Coco:

Naming:
before importing to label studio:
Name images with thunar according to <date_created>_<img_id>.jpg


Folder Format:

    dataset
        |--split
          |--train.txt # text list of all train images
          |--val.txt # text list of all validation images
          
        |--images
          |-- .jpg-files of Images (all same height and width) FORMAT: <DDMMJJ>_<IMG_ID>.jpg
          
        |--annotations
          |-- .json-file of the labelstudio-json annotations named ls_export.json

recommended command:

    cd ComputerVisionBA/Code/BT_Vision/convert_to_coco

    python lsjson_to_coco.py --dir_data test_dataset --dir_out test_dataset_coco --split_images

