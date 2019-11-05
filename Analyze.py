## -*- coding: utf-8 -*-
#"""
#Created on Mon Sep  9 15:40:03 2019
#
#@author: Omar Al Jaroudi
#"""
#
##!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import time
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt
import heartpy as hp
from signal_processing import Signal_processing
from image_processing import Image_processing
sp = Signal_processing()
ip = Image_processing()
import pickle
#
    
path = ""
cap = cv2.VideoCapture('./Offline Videos/Omar_133.mp4')
count = 0
frame_count=0
Bsig =np.array([]); Gsig =np.array([]); Rsig =np.array([]);
t0 = time.time()
times = []
bpm_old = 0
bpm=0
p1_old = 0;p2_old = 0;p3_old = 0;p4_old = 0
heartRate = []
fps = 30
PPG = np.array([])
first = 0

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

        if(count==50):
            print(1)
            t0 = time.time()
        
        if(count>=50):
            times.append(time.time() - t0)
            Bsig = np.append(Bsig,np.mean(B[np.nonzero(B)]))
            Gsig = np.append(Gsig,np.mean(G[np.nonzero(G)]))
            Rsig = np.append(Rsig,np.mean(R[np.nonzero(R)]))


            frame_count+=1
        
    cv2.imshow('Image', image)
        
    if (frame_count==300):
        if first==0:
            Gsig = sp.detrend(Gsig,10)
            Rsig = sp.detrend(Rsig,10)
            Bsig = sp.detrend(Bsig,10)
            
            Gsig = sp.normalize(Gsig)
            Rsig = sp.normalize(Rsig)
            Bsig = sp.normalize(Bsig)
            
            first = 1
            
        else:
            
            g = sp.detrend(Gsig[210:],10)
            b = sp.detrend(Bsig[210:],10)
            r = sp.detrend(Rsig[210:],10)
            
            g = sp.normalize(g)
            b = sp.normalize(b)
            r = sp.normalize(r)
            
            Gsig = Gsig[:210]
            Rsig = Rsig[:210]
            Bsig = Bsig[:210]
            
            Gsig = np.append(Gsig,g)
            Bsig = np.append(Bsig,b)
            Rsig = np.append(Rsig,r)
            
            Gsig.flatten()
            Bsig.flatten()
            Rsig.flatten()
            
        bpm,Source = sp.Analyze(Rsig,Gsig,Bsig,fps,bpm)
        PPG = np.append(PPG,Source)
        if len(heartRate)>1:
            bpm_old = heartRate[len(heartRate)-1]
            if abs(bpm_old-bpm)>30:
                bpm = bpm_old
        else:
            bpm_old = bpm
        heartRate.append(bpm*0.8+bpm_old*0.2)
        print(bpm)
        times = []
        Bsig =Bsig[90:]; Gsig = Gsig[90:]; Rsig = Rsig[90:];
        frame_count = 210
    count+=1

PPG.flatten()
with open('Running_PPG.pkl', 'wb') as f:
    pickle.dump(PPG,f)


cap.release()
cv2.destroyAllWindows()



