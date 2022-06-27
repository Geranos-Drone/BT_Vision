import os
import random
import json

img_dir = "vicon_dataset_div/images"
ls_path = "vicon_dataset_div/annotations/ls_export.json"
ls_path_old = "vicon_dataset_div/annotations/ls_export_old.json"

with open(ls_path_old) as json_file:
    data = json.load(json_file)


skipped_imgs = []
for ann in data:
    if ann["annotations"][0]["was_cancelled"]:
        skipped_imgs.append( ann["file_upload"].split('-')[1] )
        data.remove(ann)

images = os.listdir(img_dir)
print("total images: ", len(images))

images_labeled = [x for x in images if x not in skipped_imgs]


print("labeled images: ", len(images_labeled))

random.seed(42)
random.shuffle(images_labeled)
split = int(len(images_labeled)*0.8)


train_list = images_labeled[:split]
val_list = images_labeled[split:]

print("train_size: ", len(train_list))
print("val_size: ", len(val_list))


train_txt = open("vicon_dataset_div/split/train.txt", "w")
for element in train_list:
    train_txt.write(element + "\n")
train_txt.close()

val_txt = open("vicon_dataset_div/split/val.txt","w")
for element in val_list:
    val_txt.write(element + "\n")
val_txt.close()

with open(ls_path, "w") as json_file_out:
    json.dump(data, json_file_out)
