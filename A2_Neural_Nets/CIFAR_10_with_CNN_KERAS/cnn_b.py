import time
start  = time.time()
import keras   #uses tensorflow backend by default
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler

from keras.callbacks import Callback

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras.utils as utl
import os
import sys


df = pd.read_csv(sys.argv[1], delimiter = '\s+' ,header = None)

X= df.iloc[:,:-1].values
X = X.reshape(len(df), 3 ,32,32).transpose(0,2,3,1)
Y = np.array(df[len(df.columns)-1])
Y = keras.utils.to_categorical(Y.reshape(Y.shape[0], 1), 10)

X_train = X/255.0
Y_train = Y

df_test = pd.read_csv(sys.argv[2], delimiter = '\s+' ,header = None)


X_test = df_test.iloc[:,:-1].values
X_test = X_test.reshape(len(df_test), 3 ,32,32).transpose(0,2,3,1)
X_test = X_test/255.0


model = Sequential()

model.add(Conv2D(48, (3, 3), activation='relu', padding = 'same', input_shape=X_train.shape[1:]))    
model.add(BatchNormalization())

model.add(Conv2D(48, (3, 3), activation='relu', padding = 'same', strides = 2)) 
model.add(BatchNormalization())
model.add(Dropout(0.2))


model.add(Conv2D(96, (3, 3), activation='relu', padding = 'same'))    
model.add(BatchNormalization())


model.add(Conv2D(96, (3, 3), activation='relu', padding = 'same', strides = 2)) 
model.add(BatchNormalization())
model.add(Dropout(0.3))


model.add(Conv2D(192, (3, 3), activation='relu', padding = 'same', strides = 2)) 
model.add(BatchNormalization())

model.add(Conv2D(192, (3, 3), activation='relu', padding = 'same')) 
model.add(BatchNormalization())
model.add(Dropout(0.4))


model.add(Flatten())

model.add(Dense(10))
model.add(BatchNormalization())
model.add(Activation('softmax'))


batch_size = 60


class Timeout(Callback):
  def on_batch_end(self, batch, logs = None):
    if time.time() - start > 3400:
      self.model.stop_training = True

lr_reducer = keras.callbacks.ReduceLROnPlateau(monitor='acc', factor=0.5, patience= 5, verbose=1,  min_delta=0.01, min_lr=0.00005, mode='max')
end_training = Timeout()

datagen = ImageDataGenerator(rotation_range= 15,
                             width_shift_range=0.1,
                             height_shift_range = 0.1,
                             horizontal_flip=True)
datagen.fit(X_train)


rmsprop = keras.optimizers.rmsprop(lr= 0.003, decay=1e-7)
model.compile(loss='categorical_crossentropy', optimizer= rmsprop, metrics = ['accuracy'])
h = model.fit_generator(datagen.flow(X_train, Y_train, batch_size= batch_size),
                         steps_per_epoch= X_train.shape[0]//batch_size,
                         epochs= 20, 
                         workers = 4,
                         callbacks = [lr_reducer, end_training],
                         shuffle = True) 

rmsprop = keras.optimizers.rmsprop(lr= 0.002, decay=1e-7)
model.compile(loss='categorical_crossentropy', optimizer= rmsprop, metrics = ['accuracy'])
h = model.fit_generator(datagen.flow(X_train, Y_train, batch_size= batch_size),
                         steps_per_epoch= X_train.shape[0]//batch_size,
                         epochs= 20, 
                         workers = 4,
                         callbacks = [lr_reducer, end_training],
                         shuffle = True)

rmsprop = keras.optimizers.rmsprop(lr= 0.001, decay=1e-7)
model.compile(loss='categorical_crossentropy', optimizer= rmsprop, metrics = ['accuracy'])
h = model.fit_generator(datagen.flow(X_train, Y_train, batch_size= batch_size),
                         steps_per_epoch= X_train.shape[0]//batch_size,
                         epochs= 20, 
                         workers = 4,
                         callbacks = [lr_reducer, end_training],
                         shuffle = True) 

rmsprop = keras.optimizers.rmsprop(lr= 0.0005, decay=1e-7)
model.compile(loss='categorical_crossentropy', optimizer= rmsprop, metrics = ['accuracy'])
h = model.fit_generator(datagen.flow(X_train, Y_train, batch_size= batch_size),
                         steps_per_epoch= X_train.shape[0]//batch_size,
                         epochs= 20, 
                         workers = 4,
                         callbacks = [lr_reducer, end_training],
                         shuffle = True) 


Y_pred = model.predict(X_test)
y_pred = np.argmax(Y_pred, axis = 1)


with open(sys.argv[3], 'w') as f:
  for i in y_pred:
    f.write("%s\n" % i)

