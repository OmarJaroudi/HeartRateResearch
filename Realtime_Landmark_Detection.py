#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
import dlib
import numpy as np
from imutils import face_utils
import imutils
from collections import OrderedDict

LANDMARK_PTS = OrderedDict([
    ('mouth', (48, 68)),
    ('right_eyebrow', (17, 22)),
    ('left_eyebrow', (22, 27)),
    ('right_eye', (36, 42)),
    ('left_eye', (42, 48)),
    ('nose', (27, 35)),
    ('jaw', (0, 17)),
    ])


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('Face_Predictor.dat')

cap = cv2.VideoCapture(0)

while True:
   
    (ret, frame) = cap.read()
    image = frame
    
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image

    rects = detector(gray, 1)

    # loop over the face detections

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        (x, y,s,d) = cv2.boundingRect(np.array([shape[27]]))
        (a,b) = shape[35]
#        y = 2*y-b
        w = 20
        h = 8
        roi = image[y-h:y+h,x-w:x+w]
        
        roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
        cv2.imshow('ROI',roi)
        cv2.rectangle(image,(x-w,y-h),(x+w,y+h),(0,0,255),1)
        
              
            
        cv2.imshow('Image', image)
                

    if cv2.waitKey(1) == 13:  # 13 is the Enter Key
        break
cap.release()
cv2.destroyAllWindows()
