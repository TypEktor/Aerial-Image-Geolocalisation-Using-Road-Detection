
# Defining Custom Loss functions and accuracy Metrics

from keras import backend as K
import math
from keras.losses import binary_crossentropy
import tensorflow as tf

def iou_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
  union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  
  return iou

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def jacard(y_true, y_pred):

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum ( y_true_f * y_pred_f)
    union = K.sum ( y_true_f + y_pred_f - y_true_f * y_pred_f)

    return intersection/union

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def soft_dice_loss(y_true, y_pred):
   return 1-dice_coef(y_true, y_pred)

def diko_mou(y_true, y_pred, smooth=1):
  a = 0.7
  kapa = soft_dice_loss(y_true, y_pred)
  L = a*binary_crossentropy(y_true, y_pred) - (1-a)*K.log(jacard(y_true, y_pred))
  return L

def diko_mou2(y_true, y_pred, smooth=1):
  l = 30
  L = binary_crossentropy(y_true, y_pred) - l*K.log(iou_coef(y_true, y_pred))
  return L

def IoU(y_pred, y_true):
   I = tf.reduce_sum(y_pred * y_true, axis=(1, 2))
   U = tf.reduce_sum(y_pred + y_true, axis=(1, 2)) - I
   return tf.reduce_mean(I / U)