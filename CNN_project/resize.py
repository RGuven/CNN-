import os
import cv2
import numpy as np

source="./images"
dist="./resizedata"

os.mkdir(dist)

for each in os.listdir(source):
	image=cv2.imread(os.path.join(source,each))
	image=cv2.resize(image,(224,224))
	cv2.imwrite(os.path.join(dist,each),image)
