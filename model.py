
# coding: utf-8

# In[1]:


import os
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import skimage.transform
import numpy as np
import keras
from tqdm import tqdm

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.backend.tensorflow_backend import set_session
from sklearn.model_selection import train_test_split

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))


# In[2]:


from random import shuffle
ZERO_IMG_FRACTION = 0.5 #retain only a fraction of images with 0 steering angles 

#data generator
def data_gen(files, steering_angles, batch_size=32, data_dir='./data', flip=True):
    """
    Generator to yield inputs and steering angles.
    """        
    data = list(zip(files, steering_angles))

    img_id = 0    
    cnt = 0
    shuffle(data)        
    img_array = []
    y_arr = []
    
    while True:        
        if np.abs(data[img_id][1])>0 or (data[img_id][1]==0  and np.random.rand()<ZERO_IMG_FRACTION):
            # load data from file
            img_path = './data/IMG/%s'%data[img_id][0].split('/')[-1]
            img = image.load_img(img_path, target_size=(160, 320))
            
            # data preprocessing
            x = image.img_to_array(img)
            x = (x-128)/128.                
            y = data[img_id][1]

            #data augmentation
            if flip and np.random.rand()>0.5:
                x = np.fliplr(x)
                y = -y                                                                                                                                                                           

            img_array.append(x)
            y_arr.append(y)
            cnt += 1
            
        img_id += 1                    
        if img_id == len(data): 
            img_id = 0
            shuffle(data)
        
        if cnt == batch_size:
            cnt = 0
            yield np.asarray(img_array), y_arr
            img_array = []
            y_arr = []


# In[3]:


#read and augment data. Keep only a fraction of images with zero steering angle
import pandas as pd
data_csv = pd.read_csv('./data/driving_log.csv',
                      names=['center','left','right','steering_angle','throttle','break','speed'])
files = data_csv['center'].tolist()
y = data_csv['steering_angle'].tolist()


# In[4]:


# Randomly split data into train/valid set
X_train, X_test, y_train, y_test = train_test_split(files, y, test_size=0.05, random_state=42)

print("Training set: %d"%len(X_train))
print("Validation set: %d"%len(X_test))


# In[5]:


# Create train and validation set data generators
BATCH_SIZE = 512

train_generator      = data_gen(X_train, y_train, batch_size=BATCH_SIZE, data_dir='./data', flip=True)
validation_generator = data_gen(X_test, y_test, batch_size=BATCH_SIZE, data_dir='./data', flip=True)


# In[10]:


#Defining drive-net model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Cropping2D
from keras.models import Sequential, Model
from keras.layers import Lambda, Input
from keras import backend as K
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import regularizers
from keras.models import load_model
import keras

L2_REG = 1e-6

model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))

model.add(Convolution2D(16, kernel_size=(3,3), 
                        activation = 'relu', 
                        input_shape=(160, 320, 3), 
                        init='he_normal', 
                        padding='same',
                        strides=(1, 1),
                        kernel_regularizer=regularizers.l2(L2_REG)))
model.add(Convolution2D(32, kernel_size=(3,3), 
                        activation = 'relu', 
                        input_shape=(160, 320, 3), 
                        init='he_normal', 
                        padding='same',
                        strides=(1, 1),
                        kernel_regularizer=regularizers.l2(L2_REG)))

model.add(MaxPooling2D(strides=(2, 2)))

model.add(Convolution2D(32,  kernel_size=(3,3),
                        activation = 'relu', 
                        init='he_normal', 
                        padding='same',
                        strides=(1, 1),
                        kernel_regularizer=regularizers.l2(L2_REG)))
model.add(Convolution2D(64,  kernel_size=(3,3),
                        activation = 'relu', 
                        init='he_normal', 
                        padding='same',
                        strides=(1, 1),
                        kernel_regularizer=regularizers.l2(L2_REG)))
model.add(MaxPooling2D(strides=(2, 2)))

model.add(Convolution2D(64,  kernel_size=(3,3),
                        activation = 'relu', 
                        init='he_normal', 
                        padding='same',
                        strides=(1, 1),
                        kernel_regularizer=regularizers.l2(L2_REG)))
model.add(Convolution2D(128,  kernel_size=(3,3),
                        activation = 'relu', 
                        init='he_normal', 
                        padding='same',
                        strides=(1, 1), 
                        kernel_regularizer=regularizers.l2(L2_REG)))
model.add(MaxPooling2D(strides=(2, 2)))

model.add(Flatten())
model.add(Dense(256, activation = 'relu', init='he_normal', kernel_regularizer=regularizers.l2(L2_REG)))
model.add(Dropout(0.5))
model.add(Dense(128, activation = 'relu', init='he_normal', kernel_regularizer=regularizers.l2(L2_REG)))
model.add(Dropout(0.5))
model.add(Dense(1, activation = 'linear', init='he_normal'))


# In[11]:


model.summary()


# In[12]:


# Load a pretrained model, if training from a previous model
#model = load_model('model_1.h5')


# In[13]:


# Training drive-net with SGD with momentum
optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=0, nesterov=True)

model.compile(optimizer=optimizer, loss='mean_squared_error')
history = model.fit_generator(train_generator,
                              steps_per_epoch=len(X_train) /
                              BATCH_SIZE, epochs=100,
                              validation_data=validation_generator, 
                              validation_steps=4)                    
print(history.history.keys())
plt.plot(history.history['loss'][5:])
plt.plot(history.history['val_loss'][5:])
plt.title('model accuracy')
plt.ylabel('MSE')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[14]:


model.save('model_100.h5')


# In[15]:


# Fine-tuning drive-net with SGD with momentum and a smaller learning rate
optimizer = keras.optimizers.SGD(lr=0.0001, momentum=0.9, decay=0, nesterov=True)

model.compile(optimizer=optimizer, loss='mean_squared_error')
history = model.fit_generator(train_generator,
                              steps_per_epoch=len(X_train) /
                              BATCH_SIZE, epochs=100,
                              validation_data=validation_generator, 
                              validation_steps=4)                    
print(history.history.keys())
plt.plot(history.history['loss'][5:])
plt.plot(history.history['val_loss'][5:])
plt.title('model accuracy')
plt.ylabel('MSE')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[16]:


model.save('model_200.h5')

