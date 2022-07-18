import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

def VGG_19(weights_path):
    
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu',name='block1_conv1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu',name='block1_conv2'))
    model.add(MaxPooling2D((2,2), strides=(2,2),name='block1_pool'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu',name='block2_conv1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu',name='block2_conv2'))
    model.add(MaxPooling2D((2,2), strides=(2,2),name="block2_pool"))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu',name='block3_conv1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu',name='block3_conv2'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu',name='block3_conv3'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu',name='block3_conv4'))
    model.add(MaxPooling2D((2,2), strides=(2,2),name='block3_pool'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu',name='block4_conv1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu',name='block4_conv2'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu',name='block4_conv3'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu',name='block4_conv4'))
    model.add(MaxPooling2D((2,2), strides=(2,2),name='block4_pool'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu',name='block5_conv1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu',name='block5_conv2'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu',name='block5_conv3'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu',name='block5_conv4'))
    model.add(MaxPooling2D((2,2), strides=(2,2),name="block5_pool"))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu',name='fc1'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu',name='fc2'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax',name='predictions'))

    
    model.load_weights(weights_path)

    return model