import time
# t1 = time.time()
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, BatchNormalization, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator

train_file = sys.argv[1]
test_file = sys.argv[2]
out_file = sys.argv[3]

train_data = pd.read_csv(train_file, sep=' ', dtype='uint8', header=None)
train_data = np.array(train_data, dtype=np.uint8)
X = train_data[:,:-2]
y = train_data[:,-1:]

X = X.reshape(X.shape[0],3,32,32).transpose(0,2,3,1)

y = keras.utils.to_categorical(y, 100)
X = np.array(X/255, dtype=np.float32)


model_1 = Sequential()
model_1.add(Conv2D(100, (3,3), padding='same', activation='elu', input_shape=X.shape[1:], use_bias=False, kernel_initializer='he_uniform'))
model_1.add(BatchNormalization())
model_1.add(Conv2D(100, (3,3), padding='same', activation='elu', input_shape=X.shape[1:], use_bias=False, kernel_initializer='he_uniform'))
model_1.add(BatchNormalization())
model_1.add(MaxPooling2D(pool_size=(2,2)))
model_1.add(Dropout(0.2))
model_1.add(Conv2D(200, (3,3), padding='same', activation='elu', use_bias=False, kernel_initializer='he_uniform'))
model_1.add(BatchNormalization())
model_1.add(Conv2D(200, (3,3), padding='same', activation='elu', use_bias=False, kernel_initializer='he_uniform'))
model_1.add(BatchNormalization())
model_1.add(MaxPooling2D(pool_size=(2,2)))
model_1.add(Dropout(0.3))
model_1.add(Conv2D(400, (3,3), padding='same', activation='elu', use_bias=False, kernel_initializer='he_uniform'))
model_1.add(BatchNormalization())
model_1.add(Conv2D(400, (3,3), padding='same', activation='elu', use_bias=False, kernel_initializer='he_uniform'))
model_1.add(BatchNormalization())
model_1.add(MaxPooling2D(pool_size=(2,2)))
model_1.add(Dropout(0.4))
model_1.add(Flatten())
model_1.add(Dense(1000, activation='elu', use_bias=False, kernel_initializer='he_uniform'))
model_1.add(BatchNormalization())
model_1.add(Dropout(0.5))
model_1.add(Dense(100, kernel_initializer='he_uniform'))
model_1.add(Activation('softmax'))


loss_func = 'categorical_crossentropy'
batch_size = 100
total_train_time = 0

datagen = ImageDataGenerator(width_shift_range=0.15, height_shift_range=0.15, horizontal_flip=True, rotation_range=15)
datagen.fit(X)
it_train = datagen.flow(X, y, batch_size=batch_size)

opt = keras.optimizers.RMSprop(lr=0.003, decay=1e-6)
model_1.compile(loss=loss_func, optimizer=opt, metrics=['accuracy'])
model_1_log = model_1.fit_generator(it_train,
              epochs=20,
              verbose=0,
              shuffle=True, workers=8)

opt = keras.optimizers.RMSprop(lr=0.002, decay=1e-6)
model_1.compile(loss=loss_func, optimizer=opt, metrics=['accuracy'])
model_1_log = model_1.fit_generator(it_train,
              epochs=20,
              verbose=0,
              shuffle=True, workers=8)


opt = keras.optimizers.RMSprop(lr=0.001, decay=1e-6)
model_1.compile(loss=loss_func, optimizer=opt, metrics=['accuracy'])
model_1_log = model_1.fit_generator(it_train,
              epochs=20,
              verbose=0,
              shuffle=True, workers=8)

opt = keras.optimizers.RMSprop(lr=0.0005, decay=1e-6)
model_1.compile(loss=loss_func, optimizer=opt, metrics=['accuracy'])
model_1_log = model_1.fit_generator(it_train,
              epochs=20,
              verbose=0,
              shuffle=True, workers=8)

opt = keras.optimizers.RMSprop(lr=0.0002, decay=1e-6)
model_1.compile(loss=loss_func, optimizer=opt, metrics=['accuracy'])
model_1_log = model_1.fit_generator(it_train,
              epochs=20,
              verbose=0,
              shuffle=True, workers=8)

test_data = pd.read_csv(test_file, sep=' ', dtype='uint8', header=None)
test_data = np.array(test_data, dtype=np.uint8)
X_test = test_data[:,:3072]
X_test = X_test.reshape(X_test.shape[0],3,32,32).transpose(0,2,3,1)
X_test = np.array(X_test/255, dtype=np.float32)

y_hat = model_1.predict(X_test)
y_class = np.argmax(y_hat, axis=1)

o_file = open(out_file, 'w')
for i in y_class:
    o_file.write(str(i)+'\n')
o_file.close()

# print(time.time() - t1)
# model_1.save('model_sub_5.h5')
