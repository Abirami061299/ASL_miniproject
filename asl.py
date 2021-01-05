import numpy as np
import cv2 as cv
import pandas as pd



data1=pd.read_csv('/home/abirami/project/train',index_col=0)
data2=pd.read_csv('/home/abirami/project/test.csv',index_col=0)

print(data1.shape)
print(data2.shape)
data1=np.array(data1)
data2=np.array(data2)

print(data1.shape)
print(data2.shape)



X=data1[:,1:]
a=data1[ : ,0 ].reshape(2000,1)
b=data2[:,:]
Y=np.hstack((a,b))
print(X.shape)
print(Y.shape)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=6)


from keras.preprocessing.image import ImageDataGenerator



model=Sequential([])

model.add(Conv2D(300,activation="relu",input_shape=x_train.shape[1]))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

model.add(Conv2D(200,activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

model.add(Conv2D(200,activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))



model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(100,activation="relu"))
model.add(BatchNormalization())
model.add(Dense(26,activation="softmax"))

model.compile(loss="sparse_categorical_crossentropy",optimizer='adam',metrics=['accuracy'])
model.fit(traindata_generator, epoch=20,validationdata=validationdata_generator)


model.evaluate(x_test,y_test)
model.summary()

model.save('my_model.h5')
