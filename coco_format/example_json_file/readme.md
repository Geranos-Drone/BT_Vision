coco_url und flickr_url anscheinend nicht wichtig, werden nicht fuer algorithmus gebraucht:
    <Note that coco_url, flickr_url, and date_captured are just for reference. Your deep learning application probably will only need the file_name.>

annotations->keypoints stimmt noch nicht. Habe das Gefuehl, dass es die Prozentualen locations sind. Man muesste es also noch mit height respect. width multiplizieren.

Gute Doc fuers COCO dataset Format:
https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch
-> keypoint detection format

auch gut (wenn auch weniger info):

https://cocodataset.org/#format-data