# import cv2
# import numpy as np
# import pickle
#
# # cap = cv2.VideoCapture("../VID_20180605_144727.mp4") # UHD 4K
# cap = cv2.VideoCapture("../VID_20180605_151627.mp4") # 1080x720
# face_cascade = cv2.CascadeClassifier('classifiers/haarcascade_frontalface_alt2.xml')
# # recognizer = cv2.face.LBPHFaceRecognizer_create()
#
# while(True):
#     ret, frame = cap.read()
#     # frame = cv2.flip(frame, -1)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5)
#
#     for (x, y, w, h) in faces:
#         # roi_gray = gray[y:y+h, x:x+w]
#         # roi_color = frame[y:y+h, x:x+w]
#         # cv2.imshow("face", roi_color)
#         cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
#         crop_img = frame[y:y+h, x:x+w]
#         cv2.imshow('crop',crop_img)
#
#     cv2.imshow("image", cv2.resize(frame, (1080, 720)))
#
#     if(cv2.waitKey(20) & 0xFF == ord('q')):
#         break
#
# cap.release()
# cv2.destroyAllWindows()


import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('classifiers/haarcascade_frontalface_alt2.xml')
face_cascade_profile = cv2.CascadeClassifier('classifiers/haarcascade_profileface.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recognizers/face-trainner.yml")

labels = {}
with open("pickles/face-labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture("../VID_20180605_144727.mp4") # complete video UHD 4K
# cap = cv2.VideoCapture("../VID_20180606_123624.mp4") # video 4K of camilo
# cap = cv2.VideoCapture("../VID_20180606_123858.mp4") # video 4K of gallo
# cap = cv2.VideoCapture("../VID_20180606_141350.mp4") # video random
# cap = cv2.VideoCapture("../VID_20180605_151627.mp4") # complete video 1080x720
# cap = cv2.VideoCapture("../WhatsApp Video 2018-06-05 at 5.54.27 PM.mp4") # video of gallo
# cap = cv2.VideoCapture("../WhatsApp Video 2018-06-05 at 6.12.22 PM.mp4") # video of camilo
# ret = True
ret, frame = cap.read()

for i in range(150):
	ret, frame = cap.read()

cont = 0

while(ret):
	if(cont % 10 == 0):
		frame = frame[:1800, 700:3000, :]
		frame = cv2.flip(frame, -1)
		# frame = frame[400:1800, 1000:]

		# Capture frame-by-frame
		gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		band = False

		faces = face_cascade_profile.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
		for (x, y, w, h) in faces:
			roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
			roi_color = frame[y:y+h, x:x+w]
			band = True

			# recognize? deep learned model predict keras tensorflow pytorch scikit learn
			id_, conf = recognizer.predict(roi_gray)
			if conf>=40 and conf <= 120:
				print(conf, labels[id_], h, w)

				font = cv2.FONT_HERSHEY_SIMPLEX
				name = labels[id_]
				color = (255, 255, 255)
				stroke = 2
				cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

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
				# print(conf, labels[id_], h, w)
				if conf>=40 and conf <= 120:
					print(conf, labels[id_], h, w)

					font = cv2.FONT_HERSHEY_SIMPLEX
					name = labels[id_]
					color = (255, 255, 255)
					stroke = 2
					cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

				color = (255, 0, 0) #BGR 0-255
				stroke = 2
				end_cord_x = x + w
				end_cord_y = y + h
				cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

		# Display the resulting frame
		cv2.imshow('frame',cv2.resize(frame, (1080, 720)))
		#cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	cont += 1
	ret, frame = cap.read()

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
