#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 22:18:54 2017

@author: shiva
"""

import numpy as np
import pandas as pd

train_df = pd.read_json("/Users/shiva/Google Drive/Kaggle/Training/processed/train.json")
#test_df=pd.read_json("/Users/shiva/Google Drive/Kaggle/Testing/processed/test.json")


#training data

train_band_1=np.array([np.array(band).astype(np.float32).reshape(75,75) for band in train_df["band_1"]])

train_band_2=np.array([np.array(band).astype(np.float32).reshape(75,75) for band in train_df["band_2"]])

train_set=np.concatenate([train_band_1[:,:,:,np.newaxis],train_band_2[:,:,:,np.newaxis]],axis=-1)

y_train = np.array(train_df["is_iceberg"])


from keras.models import Sequential
from keras.layers import Convolution2D, GlobalMaxPooling2D, Dense, Dropout

model = Sequential()
model.add(Convolution2D(32, 5,5, activation="relu", input_shape=(75, 75, 2)))
model.add(Convolution2D(32, 3,3, activation="relu", input_shape=(75, 75, 2)))
model.add(Convolution2D(64, 3,3, activation="relu", input_shape=(75, 75, 2)))
model.add(GlobalMaxPooling2D())
model.add(Dropout(0.1))
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(1, activation="sigmoid"))
model.compile("rmsprop", "binary_crossentropy", metrics=["accuracy"])
model.summary()


model.fit(train_set, y_train, epochs=25, validation_split=0.2)

# prediction = model.predict(Y_train, verbose=1)
