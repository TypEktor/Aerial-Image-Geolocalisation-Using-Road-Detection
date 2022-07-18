# Segmentation of Road from Satellite imagery

# Importing Libraries
import warnings
warnings.filterwarnings('ignore')

import os
import cv2
from google.colab.patches import cv2_imshow
import numpy as np
import tensorflow as tf
import pandas as pd
from keras.models import Model, load_model
from skimage.morphology import label
import pickle
import tensorflow.keras.backend as K

from matplotlib import pyplot as plt
from tqdm import tqdm_notebook
import random
from skimage.io import imread, imshow, imread_collection, concatenate_images
from matplotlib import pyplot as plt
import h5py
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from datetime import datetime
from keras.optimizers import *

import Metrics as metrics
import MuliResUnet as mru
import Res_Unet as ru
import Unet as u


#%tensorflow_version 1.x

seed = 56

# Loading Data

IMAGE_HEIGHT = IMAGE_WIDTH = 256
NUM_CHANNELS = 3
image_file = 'images.h5py'
mask_file = 'masks.h5py'

hfile = h5py.File(image_file, 'r')
n1 = hfile.get('all_images')
images = np.array(n1)
print(images.shape)
hfile.close()

hfile = h5py.File(mask_file, 'r')
n1 = hfile.get('all_masks')
masks = np.array(n1)
print(masks.shape)
print("Unique elements in the train mask:", np.unique(masks))
hfile.close()

# Displaying few Samples
plt.figure(figsize=(20,16))
x, y = 5,4
for i in range(y):  
    for j in range(x):
        plt.subplot(y*2, x, i*2*x+j+1)
        pos = i*120 + j*10
        plt.imshow(images[pos])
        plt.title('Sat img #{}'.format(pos))
        plt.axis('off')
        plt.subplot(y*2, x, (i*2+1)*x+j+1)
           
        #We display the associated mask we just generated above with the training image
        plt.imshow(masks[pos])
        plt.title('Mask #{}'.format(pos))
        plt.axis('off')
        
plt.show()

masks = np.expand_dims(masks, -1)




################################################################################
################################################################################
# CALL METRICS
################################################################################
################################################################################


print(masks.shape)
print(images.shape)
# Output
# (23068, 256, 256, 1)
# (23068, 256, 256, 3)


# Spliting Data
from sklearn.model_selection import train_test_split
train_images, test_images, train_masks, test_masks = train_test_split(images, masks, test_size=0.05, random_state=seed)
del images, masks
print("TRAIN SET")
print(train_images.shape)
print(train_masks.shape)
print("TEST SET")
print(test_images.shape)
print(test_masks.shape)


# Defining Our Model
################################################################################
################################################################################
# CALL MODELS
################################################################################
################################################################################

# PER_PARAMETERS
EPOCHS = 70 
LEARNING_RATE = 0.0001
BATCH_SIZE = 16

# Initializing Callbacks

def schedlr(epoch, lr):
    print(lr)
    if tf.math.floor(epoch/2)==0:
      print('edw1')
      new_lr = lr
    else:
      print('edw2')
      temp = tf.math.floor(epoch/2)
      new_lr = lr * (0.1)* temp
    return new_lr

def step_decay(losses):
    if float(2*np.sqrt(np.array(history.losses[-1])))<0.3:
        lrate=0.01*1/(1+0.1*len(history.losses))
        momentum=0.8
        decay_rate=2e-6
        return lrate
    else:
        lrate=0.1
        return lrate

def lr_scheduler(epoch, lr):
    decay_rate = 0.1
    decay_step = 2
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr

lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=4,
                               min_lr=0.000001,
                               verbose=1,
                               epsilon=1e-4)


# Compiling the model

model = u.unet(256, 256)
optimizer = Adam(lr=LEARNING_RATE)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[iou_coef,'accuracy'])
#model.summary()


callbacks =[LearningRateScheduler(schedlr, verbose=1)]

history = model.fit(train_images,
                    train_masks/255,
                    validation_split = 0.1,
                    epochs=EPOCHS,
                    batch_size = BATCH_SIZE, callbacks = lr_reducer)


model.save('unet1.h5')