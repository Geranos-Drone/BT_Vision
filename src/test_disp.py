import openpifpaf
import cv2

capture = cv2.VideoCapture('../images/IMG_5473.png')
_, image = capture.read()
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

with openpifpaf.show.Canvas.image(image) as ax:
    pass

input("Press Enter to stop...")