# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 20:23:46 2021

@author: 20183193
"""
#%% imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

import time
#%% Preparations
date = time.strftime("%d-%m-%Y_%H-%M-%S")
path_images = '../../Images/' # navigate to ~/cource/Images from ~/cource/Github/autoencoder.py
path_models = './models/'

# the size of the images in the PCAM dataset
IMAGE_SIZE = 96
image_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)

def get_pcam_generators(base_dir, train_batch_size=32, val_batch_size=32, classes =['0'], class_mode='input'):
     # dataset parameters
     train_path = os.path.join(base_dir, 'train+val', 'train')
     valid_path = os.path.join(base_dir, 'train+val', 'valid')

     # instantiate data generators
     datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

     train_gen = datagen.flow_from_directory(train_path,
                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             batch_size=train_batch_size,
                                             classes=classes,
                                             class_mode=class_mode)

     val_gen = datagen.flow_from_directory(valid_path,
                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             batch_size=val_batch_size,
                                             classes=classes,
                                             class_mode=class_mode)

     return train_gen, val_gen

#%% define the model
# This is the dimension of the latent space (encoding space)
latent_dim = 2

encoder = Sequential([
    Flatten(input_shape=image_shape),
    Dense(192, activation='sigmoid'),
    #Dense(64, activation='sigmoid'),
    #Dense(32, activation='sigmoid'),
    Dense(latent_dim, name='encoder_output')
])

decoder = Sequential([
    #Dense(64, activation='sigmoid', input_shape=(latent_dim,)),
    #Dense(128, activation='sigmoid'),
    Dense(image_shape[0] * image_shape[1]* image_shape[2], activation='relu'),
    Reshape(image_shape)
])

autoencoder = Model(inputs=encoder.input, outputs=decoder(encoder.output))
autoencoder.compile(loss='binary_crossentropy', optimizer='adam')

#%% saving model
# save the model and weights
model_name = path_models + f'{date}_model_autoencoder_Ldim-{latent_dim}'

# save model
model_json = autoencoder.to_json() # serialize model to JSON
with open(model_name+'.json', 'w') as json_file:
    json_file.write(model_json)

# prepare to save wheights
checkpoint = ModelCheckpoint(model_name+'_w.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

#%% train model
train_gen, val_gen = get_pcam_generators(path_images)

model_history = autoencoder.fit(train_gen, epochs=1, batch_size=32, verbose=1,
                                validation_data=val_gen,
                                callbacks=checkpoint)