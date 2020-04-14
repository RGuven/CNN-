import numpy as np
from PIL import ImageGrab
import cv2
import time

last_time=time.time()
while(True):
    screen =  np.array(ImageGrab.grab(bbox=(20,20,300,300)))
    #printscreen_numpy =   np.array(printscreen_pil.getdata(),dtype='uint8')\
    #.reshape((printscreen_pil.size[1],printscreen_pil.size[0],3)) 
    
    cv2.imshow('window',cv2.cvtColor(screen,cv2.COLOR_BGR2RGB))
    print('loop took {} sconds'.format(time.time()-last_time))
    last_time=time.time()
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
    