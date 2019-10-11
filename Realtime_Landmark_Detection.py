# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:40:03 2019

@author: Omar Al Jaroudi
"""

#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
import dlib
import numpy as np
import time
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.decomposition import PCA
from signal_processing import Signal_processing
sp = Signal_processing()
from sklearn.preprocessing import StandardScaler
from math import acos,pi,sqrt
from scipy.signal import firwin
from matplotlib import pyplot as plt
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('Face_Predictor.dat')

def normalize(l):
    l = np.array(l)
    std = np.std(l,axis = 0)
    l = l-np.mean(l,axis = 0)
    l = l/std
    return l
def length(v):
    return sqrt(v[0]**2+v[1]**2)
def dot_product(v,w):
   return v[0]*w[0]+v[1]*w[1]
def determinant(v,w):
   return v[0]*w[1]-v[1]*w[0]
def inner_angle(v,w):
   cosx=dot_product(v,w)/(length(v)*length(w))
   rad=acos(cosx) # in radians
   return rad*180/pi # returns degrees
def angle_clockwise(A, B):
    inner=inner_angle(A,B)
    det = determinant(A,B)
    if det<0: #this is a property of the det. If the det < 0 then B is clockwise of A
        return inner
    else: # if the det > 0 then A is immediately clockwise of B
        return 360-inner


def unit_vector(vector):
   
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def get_landmarks(im):
    rects = detector(im, 1)

    if len(rects) > 1:
        return "error"
    if len(rects) == 0:
        return "error"
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        '''
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        '''
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

def top_lip(landmarks):
    top_lip_pts = []
    for i in range(50,53):
        top_lip_pts.append(landmarks[i])
    for i in range(61,64):
        top_lip_pts.append(landmarks[i])
    top_lip_mean = np.mean(top_lip_pts, axis=0)
    return int(top_lip_mean[:,1])

def bottom_lip(landmarks):
    bottom_lip_pts = []
    for i in range(65,68):
        bottom_lip_pts.append(landmarks[i])
    for i in range(56,59):
        bottom_lip_pts.append(landmarks[i])
    bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
    return int(bottom_lip_mean[:,1])


def eye_brows(landmarks): 
    left_eyebrow =[];
    right_eyebrow =[];
    for i in range(17,21):
        right_eyebrow.append(landmarks[i])
    for i in range(22,27):
        left_eyebrow.append(landmarks[i])
    right_eyebrow = np.squeeze(np.asarray(right_eyebrow))
    left_eyebrow = np.squeeze(np.asarray(left_eyebrow))

    return 0


def forehead(landmarks):
    landmarks= np.array(landmarks);
    if landmarks == "error":
        return [-1,-1], [-1,-1]
    #print(landmarks.shape)
    left_eye =landmarks[42];
    right_eye =landmarks[36];
    for i in range(37,41):
        right_eye=np.vstack((right_eye,landmarks[i]))
    for i in range(43,48):
        left_eye= np.vstack((left_eye,landmarks[i]))

    edge_right = right_eye[np.argmin(right_eye,0)[0],:]
    edge_left = left_eye[np.argmax(left_eye,0)[0],:]
    mean_eye = np.mean([edge_right,edge_left],0);
    
    top_lip_pts = []
    for i in range(50,53):
        top_lip_pts.append(landmarks[i])
    for i in range(61,64):
        top_lip_pts.append(landmarks[i])
    top_lip_mean = np.mean(top_lip_pts, axis=0)
    top_lip_mean = np.array(top_lip_mean);
    height = np.linalg.norm(mean_eye-top_lip_mean);
    
    leftmostEye = left_eye[np.argmin(left_eye,0)[0],:]
    rightmostEye = right_eye[np.argmax(right_eye,0)[0],:]
    distanceEyes = np.linalg.norm(leftmostEye-rightmostEye);

    eyeline = [rightmostEye[0]-leftmostEye[0],0,0];
    eyelineAct = [rightmostEye[0]-leftmostEye[0],rightmostEye[1]-leftmostEye[1],0];
    #theta = angle_between(eyeline,eyelineAct);
    theta = angle_clockwise(eyeline, eyelineAct)
    if theta>180:
        theta = 360-theta 
        theta = - theta
    #print(theta)
    theta = theta * pi/180;
    
    alpha = np.arctan(height/2/(distanceEyes/2));
    angle = alpha-theta;
    hyp = np.sqrt((height/2)**2+(distanceEyes/2)**2);
    firstPoint = mean_eye+[-hyp*np.cos(angle),-hyp*np.sin(angle)];
    hyp = np.sqrt((height/3)**2+(distanceEyes)**2);
    alpha = np.arctan(height/3/(distanceEyes));
    angle = alpha+theta;
    oppositePoint = firstPoint+[+hyp*np.cos(angle),-hyp*np.sin(angle)];
    p1 = firstPoint
    p2 = p1 + [+distanceEyes*np.cos(theta),-distanceEyes*np.sin(theta)];
    p3 = oppositePoint
    p4 = p1 + [-height/3*np.sin(theta),-height/3*np.cos(theta)]
     

    return p1,p2,p3,p4,mean_eye

def mouth_open(image):
    landmarks = get_landmarks(image)
    if landmarks == "error":
        return image, 0
    
    image_with_landmarks = annotate_landmarks(image, landmarks)
    top_lip_center = top_lip(landmarks)
    bottom_lip_center = bottom_lip(landmarks)
    lip_distance = abs(top_lip_center - bottom_lip_center)
    return image_with_landmarks, lip_distance

def landmark_image(image):
    landmarks = get_landmarks(image)
    if landmarks == "error":
        return image, ()
    image_with_landmarks = annotate_landmarks(image, landmarks)
    return image_with_landmarks,landmarks

def Stabilize(points):
    d = []
    for i in range(0,4):
        
        diff = abs((points[i]-points[i+4])/points[i+4])
        
        if diff[0]<=0.03 and diff[1]<=0.03:
            d.append(points[i+4])
        else:
            d.append((points[i]+points[i+4])/2)
    return d

def bandpass_firwin(ntaps, lowcut, highcut, fs, window='hamming'):
    nyq = 0.5 * fs
    taps = firwin(ntaps, [lowcut, highcut], nyq=nyq, pass_zero=False,
                  window=window, scale=False)
    return taps

def plotSpectrum(y,fs):
    
    xF = np.fft.fft(y)
    N = len(xF)
    xF = xF[0:int(N/2)]
    fr = np.linspace(0,fs/2,N/2)

    plt.plot(fr,abs(xF)**2)
    plt.xlim(0.6,4)
    plt.show()
    
def Analyze(Rsig,Gsig,Bsig,fps,old):

    Gsig1 = sp.detrend(Gsig,10)
    Rsig = sp.detrend(Rsig,10)
    
    Gsig1 = normalize(Gsig1)
    Rsig = normalize(Rsig)
    
    
    bandpass_filter = bandpass_firwin(128,0.7,3,fps,'hamming')
    
   
    dataset = pd.DataFrame({'Gsig':Gsig1,'Rsig':Rsig})
    features = ['Gsig','Rsig']
    data = dataset.loc[:,features].values
    dataS = StandardScaler().fit_transform(data)
    pca = PCA(n_components=2)
    pca_res = pca.fit_transform(dataS)
    
    principalDf = pd.DataFrame(data = pca_res,columns = ['PC1','PC2'])
    
    
    PC1 = principalDf.loc[:,'PC1'].values
    PC2 = principalDf.loc[:,'PC2'].values

    
    PC1 = np.convolve(PC1,bandpass_filter)
    PC2 = np.convolve(PC2,bandpass_filter)    
    
    xF = np.fft.fft(PC1)
    N = len(xF)
    xF = xF[0:int(N/2)]
    psd  = abs(xF)**2
    ps1 = np.max(psd)
    
    xF = np.fft.fft(PC2)
    N = len(xF)
    xF = xF[0:int(N/2)]
    psd  = abs(xF)**2
    ps2 = np.max(psd)
  
    Source = np.array([])
    
    if(ps1>=ps2):
        Source = PC1
    else:
        Source = PC2
        
    w = np.fft.fft(Source)
    freqs = np.fft.fftfreq(len(w))
    idx = np.argmax(np.abs(w))
    freq = freqs[idx]
    freq_in_hertz = abs(freq * fps)
    bpm = freq_in_hertz*60
    
    bpm_Gsig = AnalyzeGsig(Gsig,fps)
    if old==0:
        bpm =  (bpm_Gsig+bpm)/2
            
     
    elif abs(bpm-old)>=20:
        if abs(bpm_Gsig-old)<20:
            bpm =  bpm_Gsig
        else:
            bpm = 0.8*old + bpm*0.2
    else:
        bpm = 0.9*bpm + old*0.1
    bpm = int(bpm)
    print("bpm = " +str(bpm))
    
    
    
    return bpm
def AnalyzeGsig(Gsig,fps):
    bandpass_filter = bandpass_firwin(128,0.7,3,fps,'hamming')

    Gsig1 = sp.detrend(Gsig,10)
    Gsig1 = normalize(Gsig1)
    Gsig1 = np.convolve(Gsig1,bandpass_filter)
    w = np.fft.fft(Gsig1)
    freqs = np.fft.fftfreq(len(w))
    idx = np.argmax(np.abs(w))
    freq = freqs[idx]
    freq_in_hertz = abs(freq * fps)
    bpm3 = freq_in_hertz*60
    return bpm3
    
path = ""
cap = cv2.VideoCapture('./Offline Videos/Jad'+path+'.mp4')

frame_count=0
Bsig =np.array([]); Gsig =np.array([]); Rsig =np.array([]);
t0 = time.time()
times = []
bpm_old = 0
p1_old = 0;p2_old = 0;p3_old = 0;p4_old = 0
heartRate = []
while cap.isOpened():

    success,image = cap.read()
    if(cv2.waitKey(1)==13 or success==False):
        break
    height, width = image.shape[:2];
    start_row, start_col = int(height * .1), int(width * .2)
    end_row, end_col = int(height * .9), int(width * .8)
    image = image[start_row:end_row , start_col:end_col]
    image_landmarks,landmarks = landmark_image(image)
    imageOrg = image.copy()
    warp = image.copy();
    if landmarks != "error" and landmarks!=():
        p1,p2,p3,p4,mean_eye = forehead(landmarks)
        points = [p1,p2,p3,p4,p1_old,p2_old,p3_old,p4_old]
        points = Stabilize(points)
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
    
    if (frame_count==420):
        bpm_old = Analyze(Rsig,Gsig,Bsig,30,bpm_old)
        heartRate.append(bpm_old)
        times = []
        Bsig =Bsig[90:]; Gsig = Gsig[90:]; Rsig = Rsig[90:];
        frame_count = 330
    frame_count+=1

plt.plot(heartRate)
plt.show()
cap.release()
cv2.destroyAllWindows()
