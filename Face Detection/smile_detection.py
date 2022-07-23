import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, image = cap.read()
    faces = face_cascade.detectMultiScale(image, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x,y), (x+w, y+h), (0, 0, 255), 3)
        roi_image = image[y:y+h, x:x+w]
        smile = smile_cascade.detectMultiScale(image, 1.05, 10)
        for sx, sy, sw, sh in smile:
            cv2.rectangle(roi_image, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 3)
    cv2.imshow('smile', image)
    k = cv2.waitKey(0) & 0xff
    if k==27:
        break

cap.release()
cv2.destroyAllWindows()

