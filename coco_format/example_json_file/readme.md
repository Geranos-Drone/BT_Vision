coco_url und flickr_url anscheinend nicht wichtig, werden nicht fuer algorithmus gebraucht:
    <Note that coco_url, flickr_url, and date_captured are just for reference. Your deep learning application probably will only need the file_name.>

annotations->keypoints stimmt noch nicht. sind Prozentuale locations. Man muss es also noch mit height respect. width multiplizieren:
conversion formula:

pixel_x = x / 100.0 * original_width
pixel_y = y / 100.0 * original_height
pixel_width = width / 100.0 * original_width
pixel_height = height / 100.0 * original_height


Gute Doc fuers COCO dataset Format:
https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch
-> keypoint detection format

auch gut (wenn auch weniger info):

https://cocodataset.org/#format-data