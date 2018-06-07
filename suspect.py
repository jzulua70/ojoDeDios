import numpy as np
import cv2 as cv
import boto3
import localProcessing as local
from os import listdir
from os.path import isfile, join


crop_image = cv.imread("suspects/suspect1.jpeg")
info_path = 'info/suspect1.txt'
local.wanted("suspects/suspect2.jpeg",crop_image,info_path)

'''
region = 'us-east-1'
client=boto3.client('rekognition',region)

suspects_filePath = 'suspects/'
suspects = [join(suspects_filePath, f) for f in listdir(suspects_filePath) if isfile(join(suspects_filePath, f))]

vid = cv.VideoCapture("../VID_20180605_151627.mp4")

ret, img = vid.read()
count = 0
faces = []
while(ret):
    img = img[(len(img)*3)//8:, 0:len(img[0])*5//8]
    print(count)
    if (count %10 == 0):
        faces = local.getFaces(img)
    local.showFaces(img,faces)
    cv.waitKey(1)
    ret, img = vid.read()
    count +=1

cv.destroyAllWindows()
'''
