# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:40:03 2019

@author: Omar Al Jaroudi
"""

#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt

from signal_processing import Signal_processing
from image_processing import Image_processing
sp = Signal_processing()
ip = Image_processing()

    
path = "last"
cap = cv2.VideoCapture('./Offline Videos/Omar_'+path+'.mp4')

frame_count=0
Bsig =np.array([]); Gsig =np.array([]); Rsig =np.array([]);
t0 = time.time()
times = []
bpm_old = 0
p1_old = 0;p2_old = 0;p3_old = 0;p4_old = 0
heartRate = []
fps = cap.get(cv2.CAP_PROP_FPS)

while cap.isOpened():
    
    success,image = cap.read()
    if(cv2.waitKey(1)==13 or success==False):
        break
    height, width = image.shape[:2];
    start_row, start_col = int(height * .1), int(width * .2)
    end_row, end_col = int(height * .9), int(width * .8)
    image = image[start_row:end_row , start_col:end_col]
    image_landmarks,landmarks = ip.landmark_image(image)
    imageOrg = image.copy()
    warp = image.copy();
    if landmarks != "error" and landmarks!=():
        p1,p2,p3,p4,mean_eye = ip.forehead(landmarks)
        points = [p1,p2,p3,p4,p1_old,p2_old,p3_old,p4_old]
        points = ip.Stabilize(points)
        p1 = points[0]
        p2 = points[1]
        p3 = points[2]
        p4 = points[3]
        
        p1_old = p1
        p2_old = p2
        p3_old = p3
        p4_old = p4
        
        pts = np.array( [p1, p2, p3, p4], np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(image, [pts], True, (0,0,255), 3)
        cv2.circle(image, tuple(p1.astype(int)), 3, color=(0, 255, 255))
        cv2.circle(image, tuple(mean_eye.astype(int)), 3, color=(0, 255, 255))
        mask = np.zeros(imageOrg.shape, dtype=np.uint8)
        roi_corners = np.array([[p1, p2, p3,p4]], dtype=np.int32)
        channel_count = imageOrg.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,)*channel_count
        cv2.fillPoly(mask, roi_corners, ignore_mask_color)
        # apply the mask
        roi = cv2.bitwise_and(imageOrg, mask)
        cv2.imshow("ROI",roi)
        B, G, R = cv2.split(roi)

        if(frame_count==150):
            print(1)
            t0 = time.time()
        
        if(frame_count>=150):
            times.append(time.time() - t0)
            Bsig = np.append(Bsig,np.mean(B[np.nonzero(B)]))
            Gsig = np.append(Gsig,np.mean(G[np.nonzero(G)]))
            Rsig = np.append(Rsig,np.mean(R[np.nonzero(R)]))
            
            
    cv2.imshow('Image', image)
    
    if (frame_count==449):
        bpm = sp.Analyze(Rsig,Bsig,Gsig,fps)
        heartRate.append(bpm)
        times = []
        Bsig =Bsig[90:]; Gsig = Gsig[90:]; Rsig = Rsig[90:];
        frame_count = 359
    frame_count+=1

plt.plot(heartRate)
plt.show()
cap.release()
cv2.destroyAllWindows()
