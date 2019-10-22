# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 11:14:52 2019

@author: Omar Al Jaroudi
"""
from math import acos,pi,sqrt
import numpy as np
import dlib
import cv2

"""
This is a utility class which contains functions related to forehead detection
"""

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('Face_Predictor.dat')

class Image_processing():
    def __init__(self):
        self.a = 1
        
    def length(self,v):
        """
        @input: v 2D vector
        @function: computes the length of the vector: x^2+y^2
        @return: return the length of the 2D vector
        """
        return sqrt(v[0]**2+v[1]**2)
    def dot_product(self,v,w):
        """
        @input: 2 2D vector's v and w 
        @function: computes the dot product v.w
        @return: scaler equal to the dot product of the two vectors
        """
       return v[0]*w[0]+v[1]*w[1]
    def determinant(self,v,w):
        """
        @input: vectors v w
        @function: computes determinant
        @return: scaler equal to determinant result
        """
       return v[0]*w[1]-v[1]*w[0]
    def inner_angle(self,v,w):
        """
        @input: vectors v,w
        @function: computes the inner angle between v and w
        @return: returns the angle in degrees
        """
       cosx=self.dot_product(v,w)/(self.length(v)*self.length(w))
       rad=acos(cosx) # in radians
       return rad*180/pi # returns degrees
    def angle_clockwise(self,A, B):
        """
        @input: two vectors A and B
        @function: computes the clockwise angle between A and B
        @return: returns the clockwise angle between A and B in degrees
        """
        inner=self.inner_angle(A,B)
        det = self.determinant(A,B)
        if det<0: #this is a property of the det. If the det < 0 then B is clockwise of A
            return inner
        else: # if the det > 0 then A is immediately clockwise of B
            return 360-inner
    
    
    def unit_vector(self,vector):
        """
        @input: vector
        @function: divides vector by its norm
        @return: unit vector in that direction
        """
        return vector / np.linalg.norm(vector)
    
    def angle_between(self,v1, v2):
        """
        @input: two vectors v1 and v2
        @function: compute the angle between the 2 vectors
        @return: returns the angle bewteen them
        """
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    
    
    def get_landmarks(self,im):
        """
        @input: takes current frame im
        @function: gets the landmark locations 
        @return: a matrix containing landmark locations
        """
        rects = detector(im, 1)
    
        if len(rects) > 1:
            return "error"
        if len(rects) == 0:
            return "error"
        return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])
    
    
    def annotate_landmarks(self,im, landmarks):
        """
        @input:landmarks
        @function:draws a circle over landmarks
        @return: modified image
        """
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
        """
        @input: landmark matrix
        @function: computes the location of of the mean of the top lip
        @return: mean of top lip 
        """
        top_lip_pts = []
        for i in range(50,53):
            top_lip_pts.append(landmarks[i])
        for i in range(61,64):
            top_lip_pts.append(landmarks[i])
        top_lip_mean = np.mean(top_lip_pts, axis=0)
        return int(top_lip_mean[:,1])
    
    def bottom_lip(self,landmarks):
        """
        @input: landmark matrix
        @function: computes botton lip mean 
        @return: returns bottom lip mean
        """
        bottom_lip_pts = []
        for i in range(65,68):
            bottom_lip_pts.append(landmarks[i])
        for i in range(56,59):
            bottom_lip_pts.append(landmarks[i])
        bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
        return int(bottom_lip_mean[:,1])
    
    
    def eye_brows(self,landmarks):
        """
        @input:landmark matrix
        @function: computes the right and left eye brow postiions
        @return:nothing
        """ 
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
        """
        @input:landamark matrix
        @function: computes forehead location based on well known landmarks of previous functions
        @return: forehead coordinates
        """
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
        p2 = p1 + [+distanceEyes*np.cos(theta),-distanceEyes*np.sin(theta)];
        p3 = oppositePoint
        p4 = p1 + [-height/3*np.sin(theta),-height/3*np.cos(theta)]
         
    
        return p1,p2,p3,p4,mean_eye
    
    """
    #currently not in use
    def mouth_open(self,image):
        landmarks = self.get_landmarks(image)
        if landmarks == "error":
            return image, 0
        
        image_with_landmarks = self.annotate_landmarks(image, landmarks)
        top_lip_center = self.top_lip(landmarks)
        bottom_lip_center = self.bottom_lip(landmarks)
        lip_distance = abs(top_lip_center - bottom_lip_center)
        return image_with_landmarks, lip_distance
    """
    
    def landmark_image(self,image):
        """
        @input: image frame
        @function: computes image with landmarks and landmarks
        @error: throws error if landmarks cant be computed
        @return: returns the image with landmarks as well as landmarks seperately
        """
        landmarks = self.get_landmarks(image)
        if landmarks == "error":
            return image, ()
        image_with_landmarks = self.annotate_landmarks(image, landmarks)
        return image_with_landmarks,landmarks
    
    def Stabilize(self,points):
        """
        @input: takes two points
        @function:computes the difference between them and 
        if greater than a given threshold takes the mean of old and new
        @return: stabilization point
        """
        d = []
        for i in range(0,4):
            
            diff = abs((points[i]-points[i+4])/points[i+4])
            
            if diff[0]<=0.03 and diff[1]<=0.03:
                d.append(points[i+4])
            else:
                d.append((points[i]+points[i+4])/2)
        return d