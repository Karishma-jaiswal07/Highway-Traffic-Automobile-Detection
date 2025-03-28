import cv2
from tracker import *

cap = cv2.VideoCapture(r"G:\All Course Documents\11. Object tracking\11. Object tracking\object_tracking\highway.mp4")

# Object detection from Stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40) #MOG2 is helpfull for detecting the objects with the stable camera.

while True:
    ret, frame = cap.read() #reads the next frame data. ret tells that the image is frame is successfully readed and frame stores the actual data.

    mask = object_detector.apply(frame)

    cv2.imshow('Frame', frame) # Here the black frame respresents the background and the white frames represents the moving objects.
    cv2.imshow('Mask', mask)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()