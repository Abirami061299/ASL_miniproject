import matplotlib.pyplot as plt
import cv2
import os
import h5py
loc ='/home/abirami/Downloads/dataset/training_set/dogs/dog.9.jpg'


import numpy as np

f1 = cv2.imread(loc)
f2 = cv2.resize(f1,(100,100))
F1=np.array(f2)
F1=F1.reshape(1,30000)
print(F1)
from tensorflow.keras.models import load_model
model_d =load_model('./my_model.h5')

prediction = model_d.predict(F1)
prediction=np.array(prediction)
int_prediction=prediction.astype(int)
print(int_prediction.dtype)
print(int_prediction)

c=int_prediction[:,0]

if c==0:
  name='A'
elif c==1:
  name='B'
elif c==2:
  name='C'
elif c==3:
  name='D'
elif c==4:
  name='E'
elif c==5:
  name='F'
elif c==6:
  name='G'
elif c==7:
  name='H'
elif c==8:
  name='I'
elif c==9:
  name='J'
elif c==10:
  name='K'
elif c==11:
  name='L'
elif c==12:
  name='M'
elif c==13:
  name='N'
elif c==14:
  name='O'
elif c==15:
  name='P'
elif c==16:
  name='Q'
elif c==17:
  name='R'
elif c==18:
  name='S'
elif c==19:
  name='T'
elif c==20:
  name='U'
elif c==21:
  name='V'
elif c==22:
  name='W'
elif c==23:
  name='X'
elif c==24:
  name='Y'
elif c==25:
  name='Z'

cv2.imshow(name,f1)
cv2.waitKey(0)
cv2.destroyAllWindows()
