

import numpy as np
import cv2 as cv
import boto3
from botocore.exceptions import ClientError
client=boto3.client('rekognition','us-east-1')
face_cascade = cv.CascadeClassifier('classifiers/haarcascade_frontalface_alt2.xml')
#cap = cv.VideoCapture("../../video/DiegoWalk.mp4")
#cap = cv.VideoCapture(0)
#cap = cv.VideoCapture("../../video/VID_20180605_151627.mp4")
#cap = cv.VideoCapture("../../video/Diego.jpg")
#cap = cv.VideoCapture("assets/5.jpeg")

#cv.imshow("target", target)

def getCompare(img,target):
	source_face = []
	matches = []
	try:
			rekognition = boto3.client("rekognition", "us-east-1")
			cv.imwrite("assets/temp.jpg",img)
			source = "assets/temp.jpg"
			img1 = open(source, 'rb')
			img2 = open(target, 'rb')
			response = rekognition.compare_faces( SourceImage={'Bytes': img1.read()},TargetImage={'Bytes': img2.read()},SimilarityThreshold=80)

			source_face = response['SourceImageFace']
			matches = response['FaceMatches']
			
	except ClientError as e:
		print("no hay caras")

	return source_face,matches


cont = 0
while(True):
	ret, img = cap.read()
	#img = cv.resize(img, (1080,500)) 

	if (cont % 10 == 0):
		source_face,matches = getCompare(img,"../../video/Diego2.jpg")
		print(source_face)
		print(matches)
		if len(source_face) >0:
			height, width, channels = img.shape
			print "Source Face ({Confidence}%)".format(**source_face)
			ptop= int (source_face["BoundingBox"]["Top"] * height)
			pleft= int(source_face["BoundingBox"]["Left"] * width)
			pwidth = int(source_face["BoundingBox"]["Width"]*width)
			pheight = int(source_face["BoundingBox"]["Height"]*height)
			cv.circle(img,(pleft,ptop), 2, (0,0,255), -1)
			cv.circle(img,(pleft+pwidth,ptop+pheight), 2, (0,255,0), -1)
			crop_image = img[ptop:ptop+pwidth,pleft:pleft+pheight]
			cv.imshow("crop_image",crop_image)
			print("bounding", "left", pleft, "top", ptop, "width",pwidth, "height",pheight)

		

		

	# one match for each target face
		for match in matches:
			print "Target Face ({Confidence}%)".format(**match['Face'])
			print "  Similarity : {}%".format(match['Similarity'])
		
	cv.imshow('img',img)
	cont = cont+1

	if   cv.waitKey(1) & 0xFF == ord("q") :
		break


cv.destroyAllWindows()

