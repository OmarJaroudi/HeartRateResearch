#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
import dlib
import numpy as np
from imutils import face_utils
from collections import OrderedDict
from math import sqrt
from math import acos
from math import pi
import time
from sklearn.decomposition import PCA
from signal_processing import Signal_processing
sp = Signal_processing()
from scipy import signal
from statistics import stdev
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
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

def normalize(l):
    l = np.array(l)
    std = stdev(l)
    l = l-np.mean(l)
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
    
#cascade_path='haarcascade_frontalface_default.xml'
#cascade = cv2.CascadeClassifier(cascade_path)
detector = dlib.get_frontal_face_detector()


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
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
    top_lip_all_pts = np.squeeze(np.asarray(top_lip_pts))
    top_lip_mean = np.mean(top_lip_pts, axis=0)
    return int(top_lip_mean[:,1])

def bottom_lip(landmarks):
    bottom_lip_pts = []
    for i in range(65,68):
        bottom_lip_pts.append(landmarks[i])
    for i in range(56,59):
        bottom_lip_pts.append(landmarks[i])
    bottom_lip_all_pts = np.squeeze(np.asarray(bottom_lip_pts))
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
    max_right = right_eyebrow[np.argmax(right_eyebrow,0)[1],:]
    max_left = left_eyebrow[np.argmax(left_eyebrow,0)[1],:]

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
    top_lip_all_pts = np.squeeze(np.asarray(top_lip_pts))
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
    """
    print('height: ' + str(height));
    print('distanceEyes: ' + str(distanceEyes));
    print('angle: ' + str(angle));
    print('right eye: ' + str(rightmostEye));
    print('right eye: ' + str(leftmostEye));
    print('mean eye: ' + str(mean_eye));
    print('firstPoint: ' + str(firstPoint));
    print('oppositePoint: ' + str(oppositePoint));
    """    

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
    #cv2.imshow('Result', image_with_landmarks)
    #cv2.imwrite('image_with_landmarks.jpg',image_with_landmarks)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

# webcam
#cap = cv2.VideoCapture(0)

#video
cap = cv2.VideoCapture(0)

count = 0


bpm = 0
Bsig =np.array([]); Gsig =np.array([]); Rsig =np.array([]);Avg_signal = np.array([]);elapsed = []
   # Start time

frame_count = 0


while True:
    success,image = cap.read()
    
    height, width = image.shape[:2];
    start_row, start_col = int(height * .1), int(width * .2)
    end_row, end_col = int(height * .9), int(width * .8)
    image = image[start_row:end_row , start_col:end_col]
    image_landmarks,landmarks = landmark_image(image)
    imageOrg = image.copy()
    warp = image.copy();
    if landmarks != "error" and landmarks!=():
        p1,p2,p3,p4,mean_eye = forehead(landmarks)
        pts = np.array( [p1, p2, p3, p4], np.int32)
        #reshape our points in form  required by polylines
        pts = pts.reshape((-1,1,2))
        cv2.polylines(image, [pts], True, (0,0,255), 3)
        #cv2.rectangle(image, tuple(firstPoint.astype(int)), tuple(oppositePoint.astype(int)), (127,50,127), -1)
        cv2.circle(image, tuple(p1.astype(int)), 3, color=(0, 255, 255))
        cv2.circle(image, tuple(mean_eye.astype(int)), 3, color=(0, 255, 255))
        #ROI = imageOrg[p4.astype(int)[1]:p1.astype(int)[1],p1.astype(int)[0]:p4.astype(int)[0]]
        mask = np.zeros(imageOrg.shape, dtype=np.uint8)
        roi_corners = np.array([[p1, p2, p3,p4]], dtype=np.int32)
        # fill the ROI so it doesn't get wiped out when the mask is applied
        channel_count = imageOrg.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,)*channel_count
        cv2.fillPoly(mask, roi_corners, ignore_mask_color)
        # from Masterfool: use cv2.fillConvexPoly if you know it's convex
        # apply the mask
        roi = cv2.bitwise_and(imageOrg, mask)
        
        B, G, R = cv2.split(roi)
        if(frame_count==50):
            t = time.time()
            print(1)
        if(frame_count>=50):
            elapsed.append (time.time()-t)
            Bsig = np.append(Bsig,np.mean(B[np.nonzero(B)]))
            Gsig = np.append(Gsig,np.mean(G[np.nonzero(G)]))
            Rsig = np.append(Rsig,np.mean(R[np.nonzero(R)]))
            Avg_signal = np.append(Avg_signal,(np.mean(B[np.nonzero(B)])+np.mean(G[np.nonzero(G)])+np.mean(R[np.nonzero(R)]))/3)
            print(time.time()-t)
            
        cv2.imshow('Image', image)
    if cv2.waitKey(1) == 13 or frame_count==600:  # 13 is the Enter Key
        break
   
    frame_count+=1



fps = (frame_count-50)/elapsed[-1]

Bsig = sp.signal_detrending(Bsig)
Gsig = sp.signal_detrending(Gsig)
Rsig = sp.signal_detrending(Rsig)

Bsig = normalize(Bsig)
Gsig = normalize(Gsig)
Rsig = normalize(Rsig)


bandpass_filter = signal.firwin(128,0.8,3,window = 'hamming')



dataset = pd.DataFrame({'Gsig':Gsig,'Rsig':Rsig,'Bsig':Bsig})
features = ['Gsig','Bsig','Rsig']
data = dataset.loc[:,features].values


pca = PCA(n_components=1)
pca_res = pca.fit_transform(data)


principalDf = pd.DataFrame(data = pca_res,columns = ['principal component 1'])
PC1 = principalDf.loc[:,'principal component 1'].values
PC1 = PC1 - np.mean(PC1)
PC1 = np.convolve(PC1,bandpass_filter)

fft_of_interest, freqs_of_interest = sp.fft(PC1, fps)
max_arg = np.argmax(fft_of_interest)
bpm = freqs_of_interest[max_arg]
print("bpm = " +str(bpm))
cap.release()
cv2.destroyAllWindows()
