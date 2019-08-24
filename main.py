#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
import dlib
import numpy as np
from imutils import face_utils
import imutils
from collections import OrderedDict
import math

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
        #highliting the reference points i want to use
        image[shape[27][1],shape[27][0]] = [0,0,255]
        image[shape[23][1],shape[23][0]] = [0,0,255]
        image[shape[20][1], shape[20][0]] = [0, 0, 255]
        image[shape[29][1], shape[29][0]] = [0, 0, 255]
        ###############

        (x,y) = shape[27]
        w = abs(shape[20][0]-shape[23][0])
        h = abs(y-shape[30][1])
        y_mid = int((y + (y - h)) / 2)
        roi = image[y-h:y_mid,int(x-w/2):int(x+w/2)]
        roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
        cv2.imshow('ROI', roi)
        cv2.rectangle(image,(int(x-w/2), int(y-h)), (int(x + w/2), y_mid), (0, 0, 255), 1)#not working
        cv2.imshow('Image', image)
    if cv2.waitKey(1) == 13:  # 13 is the Enter Key
        break
cap.release()
cv2.destroyAllWindows()
