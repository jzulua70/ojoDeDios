

import numpy as np
import cv2
import boto3
import pickle
from botocore.exceptions import ClientError
client=boto3.client('rekognition','us-east-1')
face_cascade = cv2.CascadeClassifier('classifiers/haarcascade_frontalface_alt2.xml')
face_cascade_profile = cv2.CascadeClassifier('classifiers/haarcascade_profile.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recognizers/face-trainner.yml")




labels = {}
with open("pickles/face-labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}


#cap = cv2.VideoCapture("../../video/DiegoWalk2.mp4")
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("../../video/VID_20180605_151627.mp4")
#cap = cv2.VideoCapture("../../video/Diego.jpg")
#cap = cv2.VideoCapture("assets/5.jpeg")

#cv2.imshow("target", target)


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

    text = np.zeros((len(faces), (len(faces)//2), 3), np.uint8)
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

def getCompare(frame,target):
	source_face = []
	matches = []
	try:
			rekognition = boto3.client("rekognition", "us-east-1")
			cv2.imwrite("assets/temp.jpg",frame)
			source = "assets/temp.jpg"
			frame1 = open(source, 'rb')
			frame2 = open(target, 'rb')
			response = rekognition.compare_faces( SourceImage={'Bytes': frame1.read()},TargetImage={'Bytes': frame2.read()},SimilarityThreshold=80)

			source_face = response['SourceImageFace']
			matches = response['FaceMatches']
			
	except ClientError as e:
		print("no hay caras")

	return source_face,matches

cont = 0
ret, frame = cap.read()

while(ret):
	
	if(cont % 10 == 0):
		#frame = frame[:1800, 700:3000, :]
		#frame = cv2.flip(frame, -1)
		# frame = frame[400:1800, 1000:]

		# Capture frame-by-frame
		gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		band = False
		reconocido = False

		faces = face_cascade_profile.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
		for (x, y, w, h) in faces:
			roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
			roi_color = frame[y:y+h, x:x+w]
			band = True

			# recognize? deep learned model predict keras tensorflow pytorch scikit learn
			id_, conf = recognizer.predict(roi_gray)
			if conf>=40 and conf <= 140:
				print(conf, labels[id_], h, w)
				font = cv2.FONT_HERSHEY_SIMPLEX
				name = labels[id_]
				color = (255, 255, 255)
				stroke = 2
				cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
				reconocido = True

			color = (255, 0, 0) #BGR 0-255
			stroke = 2
			end_cord_x = x + w
			end_cord_y = y + h
			cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

		if not band:
			faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
			for (x, y, w, h) in faces:
				#print(x,y,w,h)
				roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
				roi_color = frame[y:y+h, x:x+w]

				# recognize? deep learned model predict keras tensorflow pytorch scikit learn
				id_, conf = recognizer.predict(roi_gray)
				#print(conf, labels[id_], h, w)
				if conf>=40 and conf <= 140:
					print(conf, labels[id_], h, w)

					font = cv2.FONT_HERSHEY_SIMPLEX
					name = labels[id_]
					color = (255, 255, 255)
					stroke = 2
					cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
					reconocido = True

				color = (255, 0, 0) #BGR 0-255
				stroke = 2
				end_cord_x = x + w
				end_cord_y = y + h
				cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)



		# Display the resulting frame
		
	#frame = cv2.resize(frame, (1080,500)) 

	if reconocido:
		source_face,matches = getCompare(frame,"../../video/Diego2.jpg")
		print(source_face)
		print(matches)
		crop_image=None
		if len(source_face) >0:
			height, width, channels = frame.shape
			print ("Source Face ({Confidence}%)".format(**source_face))
			ptop= int (source_face["BoundingBox"]["Top"] * height)
			pleft= int(source_face["BoundingBox"]["Left"] * width)
			pwidth = int(source_face["BoundingBox"]["Width"]*width)
			pheight = int(source_face["BoundingBox"]["Height"]*height)
			cv2.circle(frame,(pleft,ptop), 2, (0,0,255), -1)
			cv2.circle(frame,(pleft+pwidth,ptop+pheight), 2, (0,255,0), -1)
			crop_image = frame[ptop:ptop+pwidth,pleft:pleft+pheight]
			#cv2.imshow("crop_image",crop_image)
			print("bounding", "left", pleft, "top", ptop, "width",pwidth, "height",pheight)

	# one match for each target face
		if len(matches)>0:
			wanted("../../video/Diego2.jpg", crop_image,"info/suspect1.txt")

		for match in matches:
			print ("Target Face ({Confidence}%)".format(**match['Face']))
			print ("  Similarity : {}%".format(match['Similarity']))
	

	cv2.imshow('frame',frame)
	cont = cont+1

	if   cv2.waitKey(1) & 0xFF == ord("q") :
		break
	ret, frame = cap.read()

cv2.destroyAllWindows()

