# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 11:14:52 2019

@author: Omar Al Jaroudi
"""
from math import acos,pi,sqrt
import numpy as np
import dlib
import cv2



detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('Face_Predictor.dat')

class Image_processing():
    def __init__(self):
        self.a = 1
        
    def length(self,v):
        return sqrt(v[0]**2+v[1]**2)
    def dot_product(self,v,w):
       return v[0]*w[0]+v[1]*w[1]
    def determinant(self,v,w):
       return v[0]*w[1]-v[1]*w[0]
    def inner_angle(self,v,w):
       cosx=self.dot_product(v,w)/(self.length(v)*self.length(w))
       rad=acos(cosx) # in radians
       return rad*180/pi # returns degrees
    def angle_clockwise(self,A, B):
        inner=self.inner_angle(A,B)
        det = self.determinant(A,B)
        if det<0: #this is a property of the det. If the det < 0 then B is clockwise of A
            return inner
        else: # if the det > 0 then A is immediately clockwise of B
            return 360-inner
    
    
    def unit_vector(self,vector):
       
        return vector / np.linalg.norm(vector)
    
    def angle_between(self,v1, v2):
    
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    
    
    def get_landmarks(self,im):
        rects = detector(im, 1)
    
        if len(rects) > 1:
            return "error"
        if len(rects) == 0:
            return "error"
        return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])
    
    
    def annotate_landmarks(self,im, landmarks):
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
    
    def top_lip(self,landmarks):
        top_lip_pts = []
        for i in range(50,53):
            top_lip_pts.append(landmarks[i])
        for i in range(61,64):
            top_lip_pts.append(landmarks[i])
        top_lip_mean = np.mean(top_lip_pts, axis=0)
        return int(top_lip_mean[:,1])
    
    def bottom_lip(self,landmarks):
        bottom_lip_pts = []
        for i in range(65,68):
            bottom_lip_pts.append(landmarks[i])
        for i in range(56,59):
            bottom_lip_pts.append(landmarks[i])
        bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
        return int(bottom_lip_mean[:,1])
    
    
    def eye_brows(self,landmarks): 
        left_eyebrow =[];
        right_eyebrow =[];
        for i in range(17,21):
            right_eyebrow.append(landmarks[i])
        for i in range(22,27):
            left_eyebrow.append(landmarks[i])
        right_eyebrow = np.squeeze(np.asarray(right_eyebrow))
        left_eyebrow = np.squeeze(np.asarray(left_eyebrow))
    
        return 0
    
    
    def forehead(self,landmarks):
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
        theta = self.angle_clockwise(eyeline, eyelineAct)
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
#        p1-=(20,0)
        p2 = p1 + [+distanceEyes*np.cos(theta),-distanceEyes*np.sin(theta)];
#        p2+=(50,0)
        p3 = oppositePoint
        p4 = p1 + [-height/3*np.sin(theta),-height/3*np.cos(theta)]
#        p3+=(30,0) 
    
        return p1,p2,p3,p4,mean_eye
    
    def mouth_open(self,image):
        landmarks = self.get_landmarks(image)
        if landmarks == "error":
            return image, 0
        
        image_with_landmarks = self.annotate_landmarks(image, landmarks)
        top_lip_center = self.top_lip(landmarks)
        bottom_lip_center = self.bottom_lip(landmarks)
        lip_distance = abs(top_lip_center - bottom_lip_center)
        return image_with_landmarks, lip_distance
    
    def landmark_image(self,image):
        landmarks = self.get_landmarks(image)
        if landmarks == "error":
            return image, ()
        image_with_landmarks = self.annotate_landmarks(image, landmarks)
        return image_with_landmarks,landmarks
    
    def Stabilize(self,points):
        d = []
        for i in range(0,4):
            
            diff = abs((points[i]-points[i+4])/points[i+4])
            
            if diff[0]<=0.02 and diff[1]<=0.02:
                d.append(points[i+4])
            else:
                d.append((points[i]+points[i+4])/2)
        return d
