## Instructions for LSjson to Coco:

Folder Format:

    dataset
        |--split
          |--train.txt # text list of all train images
          |--val.txt # text list of all validation images
          
        |--images
          |-- .jpg-files of Images (all same height and width) FORMAT: <DDMMJJ>_<IMG_ID>.jpg
          
        |--annotations
          |-- .json-file of the labelstudio-json annotations

