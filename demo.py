import cv2
import numpy as np
import sys

face_cascade = cv2.CascadeClassifier("face.xml")
mouth_cascade = cv2.CascadeClassifier("mouth.xml")
smile_cascade = cv2.CascadeClassifier("smile.xml")

video_capture = cv2.VideoCapture(0)

while True:

    ret, frame = video_capture.read()
    img = frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face = face_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5, minSize=(30, 30),
                                          flags=cv2.cv.CV_HAAR_SCALE_IMAGE)

    for (x, y, w, h) in face:

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        mouth = mouth_cascade.detectMultiScale(roi_gray, scaleFactor = 1.7, minNeighbors=22, minSize=(30, 30), 
                                               flags=cv2.cv.CV_HAAR_SCALE_IMAGE)

        for (x, y, w, h) in mouth:

            cv2.rectangle(roi_color, (x, y), (x+w, y+h), (255, 0, 0), 1)

            smile = smile_cascade.detectMultiScale(roi_gray, scaleFactor = 1.7, minNeighbors=22, minSize=(25, 25), 
                                                   flags=cv2.cv.CV_HAAR_SCALE_IMAGE)    

            for (x, y, w, h) in smile:

                cv2.rectangle(roi_color, (x, y), (x+w, y+h), (5, 0, 0), 1)
            
    cv2.imshow('Smile Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
