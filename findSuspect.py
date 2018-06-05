

import numpy as np
import cv2 as cv
import boto3

client=boto3.client('rekognition','us-east-1')

face_cascade = cv.CascadeClassifier('classifiers/haarcascade_frontalface_alt2.xml')
#cap = cv.VideoCapture("../../video/VID_20180605_151627.mp4")
cap = cv.VideoCapture("../../video/Diego.jpg")
#cap = cv.VideoCapture("assets/5.jpeg")
while(True):
	# Capture frame-by-frame
	ret, img = cap.read()
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	
	for (x,y,w,h) in faces:
		cv.imwrite("assets/tmp.jpg", img[y:y+w,x:x+h])
		imageFile='assets/tmp.jpg'
		with open(imageFile, 'rb') as image:
			response = client.detect_faces(Image={'Bytes': image.read()},Attributes=['ALL'])

		for faceDetail in response['FaceDetails']:
			print("the person is: " + str(faceDetail['Emotions'][0]['Type']) + " with " + str(faceDetail['Emotions'][0]['Confidence']) + "%")
		


		cv.rectangle(img,(x-25,y-25),(x+w+25,y+h+25),(255,0,0),2)
		crop_img = img[y-25:y+h+25, x-25:x+w+25]
		cv.imshow('crop',crop_img)
		crop_img = cv.resize(crop_img, (80,80)) 

		cv.imwrite('../../video/crop.png',crop_img)

		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
	img = cv.resize(img, (1080,500)) 
	cv.imshow('img',img)

	cv.waitKey(0)


cv.destroyAllWindows()