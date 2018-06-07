import numpy as np
import cv2
import boto3
import copy
import numpy as np

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
        #print (x,y,w,h)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        crop_img = img[y:y+h, x:x+w]
        cv2.imshow('crop',crop_img)

        #roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
    cv2.imshow('img',img)

def wanted(suspect_path,crop_image,info_path):
    wanted_path = "wantedTemplate/wanted.jpeg"
    wanted_image = cv2.imread(wanted_path)
    lenghtWanted = len(wanted_image[0])
    heightWanted = len(wanted_image)

    suspect_image = cv2.imread(suspect_path)
    suspect_image = cv2.resize(suspect_image,( ((lenghtWanted*80)//100) , ((heightWanted*78)//100) ))
    lenghtSuspect = len(suspect_image[0])
    heightSuspect = len(suspect_image)

    wanted_image[int(heightWanted*.2):int(heightWanted*.2) + heightSuspect,int(lenghtWanted*.1):int(lenghtWanted*.1) + lenghtSuspect] = suspect_image

    font = cv2.FONT_HERSHEY_SIMPLEX
    crop_image = cv2.resize(crop_image,(lenghtWanted,lenghtWanted))

    cv2.putText(crop_image,'Found',(10,len(crop_image)), font, 3,(255,255,255),2,cv2.LINE_AA)
    faces = np.concatenate((wanted_image, crop_image), axis=0)

    text = np.zeros((len(faces), (len(faces)//2), 3), np.uint8)*255
    text[:] = (255,255,255)

    with open(info_path) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    pos = 0
    for line in content:
        pos+=50
        cv2.putText(text,line,(10,pos), font, 1,(0,0,0),2,cv2.LINE_AA)


    full_image = np.concatenate((faces, text), axis=1)

    cv2.imshow("found",full_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
