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



def ResUnet(inputshape):
    inputs = layers.Input(shape=inputshape)
    conv1 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    #Resnet==============
    Iput1 = Conv2D(64, 1, padding = 'same', kernel_initializer = 'he_normal')(inputs)
    Iput1 = BatchNormalization()(Iput1)
    conv1 = Add()([Iput1,conv1])
    conv1 = Activation('relu')(conv1)
    #====================
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    #Resnet==============
    Iput2 = Conv2D(128, 1, padding = 'same', kernel_initializer = 'he_normal')(pool1)
    Iput2 = BatchNormalization()(Iput2)
    conv2 = Add()([Iput2,conv2])
    conv2 = Activation('relu')(conv2)
    #====================
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    #Resnet==============
    Iput3 = Conv2D(256, 1, padding = 'same', kernel_initializer = 'he_normal')(pool2)
    Iput3 = BatchNormalization()(Iput3)
    conv3 = Add()([Iput3,conv3])
    conv3 = Activation('relu')(conv3)
    #====================
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)
    conv4 = Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    #Resnet==============
    Iput4 = Conv2D(512, 1, padding = 'same', kernel_initializer = 'he_normal')(pool3)
    Iput4 = BatchNormalization()(Iput4)
    conv4 = Add()([Iput4,conv4])
    conv4 = Activation('relu')(conv4)
    #====================
    drop4 = Dropout(0.7)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    conv5 = Conv2D(1024, 3, padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(1024, 3, padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    #Resnet==============
    Iput5 = Conv2D(1024, 1, padding = 'same', kernel_initializer = 'he_normal')(pool4)
    Iput5 = BatchNormalization()(Iput5)
    conv5 = Add()([Iput5,conv5])
    conv5 = Activation('relu')(conv5)
    #====================
    drop5 = Dropout(0.7)(conv5)
    up6 = Conv2D(512, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    up6 = BatchNormalization()(up6)
    up6 = Activation('relu')(up6)
    merge6 = concatenate([conv4,up6], axis = 3)
    conv6 = Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)
    #Resnet==============
    Iput6 = Conv2D(512, 1, padding = 'same', kernel_initializer = 'he_normal')(merge6)
    Iput6 = BatchNormalization()(Iput6)
    conv6 = Add()([Iput6,conv6])
    conv6 = Activation('relu')(conv6)
    #====================
    drop6 = Dropout(0.7)(conv6)
    up7 = Conv2D(256, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop6))
    up7 = BatchNormalization()(up7)
    up7 = Activation('relu')(up7)
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)
    #Resnet==============
    Iput7 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(merge7)
    Iput7 = BatchNormalization()(Iput7)
    conv7 = Add()([Iput7,conv7])
    conv7 = Activation('relu')(conv7)
    #====================
    
    drop7 = Dropout(0.5)(conv7)
    up8 = Conv2D(128, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop7))
    up8 = BatchNormalization()(up8)
    up8 = Activation('relu')(up8)
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)
    #Resnet==============
    Iput8 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(merge8)
    Iput8 = BatchNormalization()(Iput8)
    conv8 = Add()([Iput8,conv8])
    conv8 = Activation('relu')(conv8)
    #====================
    
    up9 = Conv2D(64, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    up9 = BatchNormalization()(up9)
    up9 = Activation('relu')(up9)
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)
    conv9 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)
    #Resnet==============
    Iput9 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(merge9)
    Iput9 = BatchNormalization()(Iput9)
    conv9 = Add()([Iput9,conv9])
    conv9 = Activation('relu')(conv9)
    #====================
    
    conv9 = Conv2D(2, 3, padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
    model = Model(inputs = inputs, outputs = conv10)

   # model.compile(optimizer = Adam(lr = 2e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    #model.summary()

    return model