import numpy as np
import cv2
import boto3

contFrame=0
client=boto3.client('rekognition','us-east-1')
emotion=[]
band = False

def getFaces(img):
	global emotion,band,positionsx,positionsy

	face_cascade = cv2.CascadeClassifier('classifiers/haarcascade_frontalface_alt2.xml')

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	emotions=[]

	cont = 0
	if(len(faces)>0):
		if(contFrame%40==0):
			emotion=[]
			positionsx = []
			positionsy = []
			for (x,y,w,h) in faces:
				cv2.imwrite("assets/tmp.jpg", img[y:y+w,x:x+h])
				imageFile='assets/tmp.jpg'
				bucket='bucket'

				with open(imageFile, 'rb') as image:
						 response = client.detect_faces(Image={'Bytes': image.read()},Attributes=['ALL'])

				for faceDetail in response['FaceDetails']:
					print("the person is: " + str(faceDetail['Emotions'][0]['Type']) + " with " + str(faceDetail['Emotions'][0]['Confidence']) + "%")
					a = faces[cont].tolist()
					a.append(str(faceDetail['Emotions'][0]['Type']))
					positionsx.append(x)
					positionsy.append(y)
					emotions.append(a)
					emotion.append(str(faceDetail['Emotions'][0]['Type']))
					band=True
				cont += 1
		else:
			contEm=0
			if len(faces) <= len(emotion):
				for (x,y,w,h) in faces:
					a = faces[contEm].tolist()
					bestDist = abs(positionsx[0] - x) + abs(positionsy[0]-y)
					bestMatch = 0
					for i in range(len(positionsx)):
						xvar = positionsx[i]
						yvat = positionsy[i]
						if (bestDist > abs (positionsx[i] - x) + abs(positionsy[i] - y)):
							bestMatch = i
							bestDist = abs (positionsx[i] - x) + abs(positionsy[i] - y)
					a.append(emotion[bestMatch])
					emotions.append(a)
					contEm += 1

	return emotions


cap = cv2.VideoCapture(0)
while(True):
	# Capture frame-by-frame
	ret, faceOriginal = cap.read()

	#faceOriginal = cv2.imread('assets/R1.jpeg')

	rows = 500
	columns = 500

	faceResized = cv2.resize(faceOriginal, (rows, columns))

	faceCoordinates = getFaces(faceResized)

	print(faceCoordinates)
	blank_emoji = np.zeros((rows,columns,3), np.uint8)
	emojiMask = np.zeros((rows,columns,3), np.uint8)
	for (x,y,w,h,e) in faceCoordinates:
		emojiOriginal = cv2.imread('emojisExample/' + e+'.png')

		x_diff = w
		y_diff = h

		base_x = x
		base_y = y
		emojiResized = cv2.resize(emojiOriginal, (x_diff,y_diff))

		blank_emoji[base_y:base_y+y_diff,base_x:base_x+x_diff] = emojiResized

		greyEmoji = cv2.cvtColor(blank_emoji, cv2.COLOR_RGB2GRAY)

		ret,threshold = cv2.threshold(greyEmoji,50,255,0)

		RGBgray = cv2.cvtColor(threshold, cv2.COLOR_GRAY2RGB)
		notRGBgray = cv2.bitwise_not(RGBgray)

		emojiMask = cv2.bitwise_and(RGBgray,blank_emoji)
		blank_emoji = emojiMask

		backgroundMask = cv2.bitwise_and(notRGBgray, faceResized)
		faceResized = backgroundMask

	fullmask = cv2.bitwise_or(emojiMask, faceResized)

		#cv2.imshow('face',faceResized)
		#cv2.imshow('emoji',blank_emoji)
	cv2.imshow('mask',fullmask)

	contFrame+=1

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
