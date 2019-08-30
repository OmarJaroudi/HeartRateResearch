#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
import dlib
import numpy as np
from imutils import face_utils
import imutils
from collections import OrderedDict
#from matplotlib import pyplot as plt
import time
from signal_processing import Signal_processing
from statistics import mean
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
Bsig =np.array([]); Gsig =np.array([]); Rsig =np.array([]);Ysig = np.array([]);Cbsig = np.array([]); Crsig = np.array([]);

cap = cv2.VideoCapture(0)
sp = Signal_processing()
data_buffer = []
fft_of_interest = []
freqs_of_interest = []
filtered_data = []
frame_count = 0
BUFFER_SIZE = 100
Heart_rate = []
Avg = 0
L = 0
t = time.time()
fps = 0
times = []
bpm = 0
total =0
while True:
    t0 = time.time()
    frame_count+=1
    if frame_count%3==0:
        continue
    (ret, frame) = cap.read()
    image = frame

    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image

    rects = detector(gray, 1)
    
    # loop over the face detections
    count = 0
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        #highliting the reference points i want to use
        image[shape[27][1],shape[27][0]] = [0,0,255]
        image[shape[23][1],shape[23][0]] = [0,0,255]
        image[shape[20][1], shape[20][0]] = [0, 0, 255]
        image[shape[29][1], shape[29][0]] = [0, 0, 255]
        ##############
        x = int((shape[21][0]+shape[22][0])/2)
        y = int((shape[21][1]+shape[22][1])/2)
        w = abs(shape[39][0]-shape[42][0])
        h = abs(y-shape[29][1])
        y_mid = int((y + (y - h)) / 2)
        roi = image[y-h:y_mid,int(x-w/2):int(x+w/2)]
        if(roi.shape[0] !=0 and roi.shape[1] != 0):
            roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
            green = sp.extract_color(roi)
            if(abs(green-np.mean(data_buffer))>30 and L>99): #remove sudden change, if the avg value change is over 10, use the mean of the data_buffer
                green = data_buffer[-1]
        
            data_buffer.append(green)
            times.append(time.time() - t)
        L = len(data_buffer)
        if L > BUFFER_SIZE:
            data_buffer = data_buffer[-BUFFER_SIZE:]
            times = times[-BUFFER_SIZE:]
            L = BUFFER_SIZE
        if L==100:
            fps = float(L) / (times[-1] - times[0])
            detrended_data = sp.signal_detrending(data_buffer)
            interpolated_data = sp.interpolation(detrended_data, times)
            normalized_data = sp.normalization(interpolated_data)
            nyq = 0.5*fps
            l = 0.8/nyq
            high = 3/nyq
            if(l>0 and l<1 and high>0 and high<1 and high>l):
                filtered_data = sp.butter_bandpass_filter(interpolated_data, 0.8, 3, fps, order = 3)
            fft_of_interest, freqs_of_interest = sp.fft(filtered_data, fps)
            max_arg = np.argmax(fft_of_interest)
            bpm = freqs_of_interest[max_arg]
          
#            with open("a.txt",mode = "a+") as f:
#                f.write("time: {0:.4f} ".format(times[-1]) + ", HR: {0:.2f} ".format(bpm) + "\n")       
            Heart_rate.append(bpm)
            
        cv2.rectangle(image,(int(x-w/2), int(y-h)), (
                int(x + w/2), y_mid), (0, 0, 255), 1)
        cv2.imshow('Image', image)
    if cv2.waitKey(1) == 13:  # 13 is the Enter Key
        break

Avg = mean(Heart_rate)
print(Avg)
cap.release()
cv2.destroyAllWindows()
