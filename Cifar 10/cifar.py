#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 19:32:05 2017

@author: shiva
"""

import pandas as pd
import numpy as np
import cPickle

def unpickle(file):
    with open(file,'rb') as fo:
        dict=cPickle.load(fo)
    return dict

    
test_1=unpickle('data_batch_1')

data=test_1['data']
label=test_1['labels']

red_channel=data[:,0:1024]
green_channel=data[:,1024:2048]
blue_channel=data[:,2048:3072]


red_ch=np.zeros((10000,32,32))
blue_ch=np.zeros((10000,32,32))
green_ch=np.zeros((10000,32,32))
for i in range(10000):
    temp=red_channel[i,:]
    temp=np.reshape(temp,(32,32))
    red_ch[i,:,:]=temp
    
for i in range(10000):
    temp=green_channel[i,:]
    temp=np.reshape(temp,(32,32))
    green_ch[i,:,:]=temp
    
for i in range(10000):
    temp=blue_channel[i,:]
    temp=np.reshape(temp,(32,32))
    blue_ch[i,:,:]=temp
    
train_set=np.concatenate([red_ch[:,:,:,np.newaxis],green_ch[:,:,:,np.newaxis],blue_ch[:,:,:,np.newaxis]],axis=-1)

train_set/=255

from keras.models import Sequential
from keras.layers import Convolution2D, GlobalMaxPooling2D, Dense, Dropout
import keras
from keras import regularizers
one_hot_labels = keras.utils.to_categorical(label, num_classes=10)
model = Sequential()
model.add(Convolution2D(32, 3,3, activation="relu", input_shape=(32, 32, 3)))
model.add(Convolution2D(64, 3,3, activation="relu", input_shape=(32, 32, 3)))
# model.add(Convolution2D(64, 3,3, activation="relu", input_shape=(32, 32, 3)))
model.add(GlobalMaxPooling2D())
model.add(Dense(256, activation="relu",kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.001)))
model.add(Dropout(0.1))
model.add(Dense(256, activation="relu",kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.1))
model.add(Dense(128, activation="relu",kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.1))
model.add(Dense(10, activation="softmax"))
model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
model.summary()

model.fit(train_set, one_hot_labels, epochs=100, validation_split=0.15)









