import numpy as np
import cv2

girisverisi=np.array([])

for i in range(200):
	i=i+1
	uzantı="./latestimagefolder/%s.jpg"%i
	resim=cv2.imread(uzantı)
	girisverisi=np.append(girisverisi,resim)
#print(girisverisi.shape)  =>(19568640,)

girisverisi=girisverisi.reshape(200,224,224,3)
##print(girisverisi.shape)  =>(200, 224, 224, 3)

np.save("girisverisi",girisverisi)
