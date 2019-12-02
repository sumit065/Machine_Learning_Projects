import keras   #uses tensorflow backend by default
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import SGD

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras.utils as utl
import os
import sys

df = pd.read_csv(sys.argv[1], delimiter = '\s+' ,header = None)

Xr = np.array(df.iloc[:, 0:1024]).reshape([len(df), 32,32])/255.0
Xg = np.array(df.iloc[:,1024:2048]).reshape([len(df), 32,32])/255.0
Xb = np.array(df.iloc[:,2048:3072]).reshape([len(df), 32,32])/255.0
X_train = np.stack((Xr,Xg,Xb), axis = -1)

Y_train = np.array(pd.get_dummies(df[len(df.columns)-1]))

df_test = pd.read_csv(sys.argv[2], delimiter = '\s+' ,header = None)
X_test = df_test.iloc[:,:-1].values
X_test = X_test.reshape(len(df_test), 3 ,32,32).transpose(0,2,3,1)
X_test = X_test/255.0

model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape= X_train.shape[1:] , padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(BatchNormalization())
model.add(Activation('softmax'))

rmsprop = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss='categorical_crossentropy', optimizer= rmsprop, metrics = ['accuracy'])

model.fit(X_train, Y_train, batch_size= 32, validation_split = 0.1, shuffle = True,  epochs= 10 )

Y_pred = model.predict(X_test)
y_pred = np.argmax(Y_pred, axis = 1)


with open(sys.argv[3], 'w') as f:
  for i in y_pred:
    f.write("%s\n" % i)


