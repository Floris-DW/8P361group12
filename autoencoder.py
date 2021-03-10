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
from tensorflow.keras.layers import Dense, Flatten, Reshape, Conv2D, MaxPool2D, UpSampling2D, Input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

import time
#%% Preparations
date = time.strftime("%d-%m-%Y_%H-%M-%S")
path_images = '../../Images/' # navigate to ~/source/Images from ~/source/Github/autoencoder.py
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
filters = [64, 32, 16]
kernel_size=(3,3)
pool_size=(2,2)
latent_dim = filters[-1]

# encoder
input_layer = Input(shape=image_shape)
encoded_layer1 = Conv2D(filters[0], kernel_size, activation = 'relu', padding = 'same')(input_layer)
encoded_layer1 = MaxPool2D(pool_size = pool_size, padding = 'same')(encoded_layer1)
encoded_layer2 = Conv2D(filters[1], kernel_size, activation = 'relu', padding = 'same')(encoded_layer1)
encoded_layer2 = MaxPool2D(pool_size = pool_size, padding = 'same')(encoded_layer2)
encoded_layer3 = Conv2D(filters[2], kernel_size, activation = 'relu', padding = 'same')(encoded_layer2)
latent_view = MaxPool2D(pool_size = pool_size, padding = 'same')(encoded_layer3)

# decoder
decoded_layer1 = Conv2D(filters[2], kernel_size, activation = 'relu', padding = 'same')(latent_view)
decoded_layer1 = UpSampling2D(pool_size)(decoded_layer1)
decoded_layer2 = Conv2D(filters[1], kernel_size, activation = 'relu', padding = 'same')(decoded_layer1)
decoded_layer2 = UpSampling2D(pool_size)(decoded_layer2)
decoded_layer3 = Conv2D(filters[0], kernel_size, activation = 'relu', padding = 'same')(decoded_layer2)
decoded_layer3 = UpSampling2D(pool_size)(decoded_layer3)
output_layer = Conv2D(image_shape[2], kernel_size, activation = 'relu', padding = 'same')(decoded_layer3)

# compile model
autoencoder = Model(input_layer, output_layer)
autoencoder.compile(loss='MeanSquaredError', optimizer='adam')
autoencoder.summary()

#%% saving model
# save the model and weights
num_epochs = 4

model_name = path_models + f'{date}_model_autoencoder_Ldim-{latent_dim}_epochs-{num_epochs}'

# save model
model_json = autoencoder.to_json() # serialize model to JSON
with open(model_name+'.json', 'w') as json_file:
    json_file.write(model_json)

# prepare to save wheights
checkpoint = ModelCheckpoint(model_name+'_w.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

#%% train model
train_gen, val_gen = get_pcam_generators(path_images)

model_history = autoencoder.fit(train_gen, epochs=num_epochs, batch_size=32, verbose=1,
                                validation_data=val_gen,
                                callbacks=checkpoint)