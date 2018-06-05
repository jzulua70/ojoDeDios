

import numpy as np
import cv2 as cv

face_cascade = cv.CascadeClassifier('classifiers/haarcascade_frontalface_alt2.xml')
cap = cv.VideoCapture("../../video/VID_20180605_151627.mp4")
while(True):
	# Capture frame-by-frame
	ret, img = cap.read()
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
		cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		crop_img = img[y:y+h, x:x+w]
		cv.imshow('crop',crop_img)

		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
	img = cv.resize(img, (1080,500)) 
	cv.imshow('img',img)

	cv.waitKey(1)


cv.destroyAllWindows()