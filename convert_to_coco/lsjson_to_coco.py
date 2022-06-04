#!/usr/bin/env python
# coding: utf-8

# In[13]:


"""
Convert json files in labelstudio json format into json file with COCO keypoint format
"""
import glob
import os
import time
from shutil import copyfile
import json
import argparse
import re
import shutil

import numpy as np
from PIL import Image

#import matplotlib.pyplot as plt

POLE_KEYPOINTS = [
    'tip',              #1
    'top_left',         #2
    'top_right',        #3
    'mid_left',         #4
    'mid_right',        #5
    'bottom_left',      #6
    'bottom_right',     #7
]

POLE_SKELETON = [
    [1,2],[1,4],[1,5],[2,3],[2,6],[3,4],[3,7],[4,8],[5,8],[5,6],[6,7],[7,8]
]

def cli():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir_data', default='data-poles/train',
                        help='dataset directory')
    parser.add_argument('--dir_out', default='data-poles',
                        help='where to save annotations and files')
    
    parser.add_argument('--sample', action='store_true',
                        help='Whether to only process the first 50 images')
    parser.add_argument('--single_sample', action='store_true',
                        help='Whether to only process the first image')
    
    parser.add_argument('--split_images', action='store_true',
                        help='Whether to copy images into train val split folder')
    parser.add_argument('--histogram', action='store_true',
                        help='Whether to show keypoints histogram')
    args = parser.parse_args()
    return args



class LSJsonToCoco:
    
    
    sample = False
    single_sample = False
    split_images = False
    histogram = False
    
    def __init__(self, dir_dataset, dir_out):
        
        assert os.path.isdir(dir_dataset), 'dataset directory not found'
        self.dir_dataset = dir_dataset
        
        
        self.dir_out_im = os.path.join(dir_out, 'images')
        self.dir_out_ann = os.path.join(dir_out, 'annotations')
        os.makedirs(self.dir_out_im, exist_ok=True)
        os.makedirs(self.dir_out_ann, exist_ok=True)
        
        self.coco_file = {}
        
        # Load train val split
        path_train = os.path.join(self.dir_dataset, 'split', 'train.txt') #train-list.text: list of train images
        path_val = os.path.join(self.dir_dataset, 'split', 'val.txt') #validation-list.text: list of validation images
        self.splits = {}
        for name, path in zip(('train', 'val'), (path_train, path_val)): #zip creates an iterator over all input iterables
            with open(path, "r") as ff:
                lines = ff.readlines()
            self.splits[name] = [os.path.join(self.dir_dataset, 'images', line.strip())
                                 for line in lines]
            assert self.splits[name], "specified path is empty"
        
        
    def process(self):
        img_ids = []
        annotation_id = 10000
        
        data = self._load_lsjson(os.path.join(self.dir_dataset,'annotations','ls_export.json'))
        
        for phase, im_paths in self.splits.items():
            
            cnt_images = 0
            cnt_instances = 0
            cnt_kps = [0] * len(POLE_KEYPOINTS)
            self.initiate_coco()  # Initiate coco json file at each phase
            
            if self.sample:
                im_paths = im_paths[:50]
            if self.split_images:
                path_dir = (os.path.join(self.dir_out_im, phase))
                os.makedirs(path_dir, exist_ok=True)
                assert not os.listdir(path_dir), "Directory to save images is not empty. "                     "Remove flag --split_images ?"
            elif self.single_sample:
                im_paths = self.splits['train'][:1]
                print(f'Single sample for train/val:{im_paths}')
            
            for im_path in im_paths:
                im_size, im_name, im_id = self._process_image(im_path,phase)
                cnt_images += 1
                
                annotation_id += 1
                self._process_annotation(data, im_size, im_id, cnt_kps, annotation_id)
                cnt_instances += 1
            
                if self.split_images:
                    dst = os.path.join(self.dir_out_im, phase, os.path.split(im_path)[-1])
                    copyfile(im_path, dst)

                # Count
                if (cnt_images % 1000) == 0:
                    text = ' and copied to new directory' if self.split_images else ''
                    print(f'Parsed {cnt_images} images' + text)
                
        
            self.save_coco_files(phase)
            print(f'\nPhase:{phase}')
            print(f'Average number of keypoints labelled: {sum(cnt_kps) / cnt_instances:.1f} / {len(POLE_KEYPOINTS)}')
            print(f'COCO files directory:  {self.dir_out_ann}')
            print(f'Saved {cnt_instances} instances over {cnt_images} images ')
            if self.histogram:
                histogram(cnt_kps)    
        
        
    def initiate_coco(self):
        """
        Initiate Coco File for Training and validation phase
        """
        for co_file in [(self.coco_file)]:
            co_file["info"] = dict(url="https://github.com/openpifpaf/openpifpaf",
                                  date_created=time.strftime("%a, %d %b %Y %H:%M:%S +0000",
                                                             time.localtime()),
                                  description=("Conversion of Label-Studio Pole dataset into MS-COCO format"))

            skel = POLE_SKELETON
            kps = POLE_KEYPOINTS
            co_file["categories"] = [dict(name='pole',
                                         id=420,
                                         skeleton=skel,
                                         supercategory='poles',
                                         keypoints=kps)]
            co_file["images"] = []
            co_file["annotations"] = []
            
            
    def save_coco_files(self, phase):
        for co_file, n_kps in [(self.coco_file, len(POLE_KEYPOINTS))]:
            name = 'pole_keypoints_' + str(n_kps) + '_'
            if self.sample:
                name = name + 'sample_'
            elif self.single_sample:
                name = name + 'single_sample_'

            path_coco = os.path.join(self.dir_out_ann, name + phase + '.json')
            with open(path_coco, 'w') as outfile:
                json.dump(co_file, outfile)
            
        
    def _process_image(self, im_path, phase):
        """Update image field in json file"""
        file_name = os.path.basename(im_path)
        im_name = os.path.splitext(file_name)[0] # 
        im_id = int(im_name.split(sep='_')[1])  # Numeric code in the image
        im = Image.open(im_path)
        width, height = im.size
        dict_img = {
            'coco_url': "unknown",
            'file_name': file_name,
            'id': im_id,
            'license': 1,
            'date_captured': "unknown",
            'width': width,
            'height': height}
        self.coco_file["images"].append(dict_img)

        return (width, height), im_name, im_id
        
    
    def _process_annotation(self, data, im_size, im_id, cnt_kps, annotation_id):
        """Process single instance of annotations"""
        
        for i_1 in range(len(data)):
            if int(data[i_1]['file_upload'].split(sep = '_')[-1].split(sep = ".")[0]) == im_id:
                
                keypoints_coco = []
                bbox_inst = []
                area_inst = []
                annotation_inst = data[i_1].get('annotations')
                result_inst = annotation_inst[0].get('result')
                i_2 = 0
                x_i = []
                y_i = []
                num_keypoints = 0
                im_size = [result_inst[0]['original_width'],result_inst[0]['original_height']]
                for label_index, label_inst in enumerate(POLE_KEYPOINTS):
                    if (i_2 < len(result_inst)) and (result_inst[i_2]['value']['keypointlabels'][0] == label_inst):
                        x_ii = result_inst[i_2]['value']['x']*im_size[0]/100
                        y_ii = result_inst[i_2]['value']['y']*im_size[1]/100
                        keypoints_coco.append(x_ii)
                        keypoints_coco.append(y_ii)
                        x_i.append(x_ii)
                        y_i.append(y_ii)
                        keypoints_coco.append(1)
                        cnt_kps[label_index] += 1
                        num_keypoints += 1

                        i_2 += 1
                    else:
                        keypoints_coco.append(0)
                        keypoints_coco.append(0)
                        keypoints_coco.append(0)

                box_tight = [np.min(x_i), np.min(y_i), np.max(x_i), np.max(y_i)]
                w, h = box_tight[2] - box_tight[0], box_tight[3] - box_tight[1]
                x_o = max(box_tight[0] - 0.1 * w, 0)
                y_o = max(box_tight[1] - 0.1 * h, 0)
                x_i = min(box_tight[0] + 1.1 * w, im_size[0])
                y_i = min(box_tight[1] + 1.1 * h, im_size[1])
                bbox_inst = [int(x_o), int(y_o), int(x_i - x_o), int(y_i - y_o)]  # (x, y, w, h)

                
                dict_ann ={
                    "segmentation":[],
                    "num_keypoints": num_keypoints,
                    "iscrowd": 0,
                    "keypoints": keypoints_coco,
                    "image_id": im_id,
                    "category_id": 420,
                    "id": annotation_id,
                    "area": bbox_inst[2]*bbox_inst[3],
                    "bbox": bbox_inst}
                self.coco_file["annotations"].append(dict_ann)
        
        return cnt_kps
    
    def _load_lsjson(self, path):
        with open(path) as json_file:
            data_ = json.load(json_file)

               
        for it in range(len(data_)):
            img_name_ = data_[it]['file_upload'].split(sep='_')[1]
            img_id_ = int(img_name_.split(sep='.')[0])
            data_[it]['id'] = img_id_
            data_[it]['annotations'][0]['id'] = img_id_

            
        return data_

def histogram(cnt_kps):
    bins = np.arange(len(cnt_kps))
    data = np.array(cnt_kps)
    plt.figure()
    plt.title("Distribution of the keypoints")
    plt.bar(bins, data)
    plt.xticks(np.arange(len(cnt_kps), step=5))
    plt.show()
    

def main():
    
    args = cli()
    LSJsonToCoco.sample = args.sample
    LSJsonToCoco.single_sample = args.single_sample
    LSJsonToCoco.split_images = args.split_images
    LSJsonToCoco.histogram = args.histogram
    
    
    lsjson_coco = LSJsonToCoco(args.dir_data, args.dir_out)
    lsjson_coco.process()
    
    
    
if __name__ == "__main__":
    main()

"""path = os.path.join('test_dataset','annotations','ls_export.json')

with open(path) as json_file:
    data_ = json.load(json_file)


for it in range(len(data_)):
    img_name_ = data_[it]['file_upload'].split(sep='_')[1]
    img_id_ = int(img_name_.split(sep='.')[0])
    data_[it]['id'] = img_id_
    data_[it]['annotations'][0]['id'] = img_id_

data = data_
im_id = 1002

for i_1 in range(len(data)):
    print(data[i_1]['file_upload'].split(sep = '_')[-1].split(sep = ".")[0])
    if int(data[i_1]['file_upload'].split(sep = '_')[-1].split(sep = ".")[0]) == im_id:
        print("1002")"""


# In[ ]:




