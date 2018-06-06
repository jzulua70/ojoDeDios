import numpy as np
import cv2
import boto3
import copy
from os import listdir
from os.path import isfile, join


def getFaces(img):
    face_cascade = cv2.CascadeClassifier('classifiers/haarcascade_frontalface_alt2.xml')
    profile_cascade = cv2.CascadeClassifier('classifiers/haarcascade_profile.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces_profile = profile_cascade.detectMultiScale(gray, 1.3, 5)
    faces_front   = face_cascade.detectMultiScale(gray,1.3, 5)
    faces = [x for x in faces_profile]
    faces += [x for x in faces_front]
    return faces

def showFaces(img, faces):
    for (x,y,w,h) in faces:
        print (x,y,w,h)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        crop_img = img[y:y+h, x:x+w]
        cv2.imshow('crop',crop_img)

        #roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
    cv2.imshow('img',img)
