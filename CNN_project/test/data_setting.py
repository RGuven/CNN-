import numpy as np
import cv2

girisverisi=np.array([])

for i in range(5):
	i=i+1
	uzantı="./test_latestimagefolder/%s.jpg"%i
	resim=cv2.imread(uzantı)
	girisverisi=np.append(girisverisi,resim)
#print(girisverisi.shape)  =>(19568640,)

girisverisi=girisverisi.reshape(5,224,224,3)
##print(girisverisi.shape)  =>(130, 224, 224, 3)

np.save("test_girisverisi",girisverisi)
