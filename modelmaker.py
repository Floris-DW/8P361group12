# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 20:23:46 2021

@author: 20183193
"""
version = 'AE_v1' # naming sceme for models, prevent namingconflicts. AE = AutoEncoder

#%% imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D, Input
from tensorflow.keras.callbacks import ModelCheckpoint

import time
#%% Preparations
date = time.strftime("%d-%m-%Y_%H-%M-%S")
path_images = '../../Images/' # navigate to ~/source/Images from ~/source/Github/autoencoder.py
path_models = './models/'

# the size of the images in the PCAM dataset
IMAGE_SIZE = 96
image_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)


def AutoEncoder(input_shape=(96,96,3), filters=[64, 32, 16], kernel_size=(3,3), pool_size=(2,2),
                activation='relu',padding='same',
                loss='MeanSquaredError', optimizer='adam',
                model_name=None):
    x = Input(shape=input_shape)
    #encoder
    for i in filters:
        x = Conv2D(i, kernel_size, activation=activation, padding=padding)(x)
        x = MaxPool2D(pool_size=pool_size, padding=padding)(x)
    x = MaxPool2D(pool_size=pool_size, padding=padding)(x) # latent vieuw

    # decoder
    for i in filters[::-1]: #loop over the filter in reverse
        x = Conv2D(filters[2], kernel_size, activation=activation, padding=padding)(x)
        x = UpSampling2D(pool_size)(x)
    x = Conv2D(input_shape[2], kernel_size, activation=activation, padding=padding)(x) # output layer

    if model_name==None:
        model_name=f'{version}_F{filters}_K{kernel_size}_P{pool_size}'

    model = Model(Input(shape=input_shape), x, name=model_name)
    model.compile(optimizer,loss)
    return model


def TrainModel(model, train, validation, num_epochs, batch_size=32, save_model=True, verbose=1, save_dir='./models/'):
    if save_model == True:
        model_json = model.to_json() # serialize model to JSON
        with open(model.name+'.json', 'w') as json_file:
            json_file.write(model_json)

        # prepare to save wheights
        date = time.strftime("%d-%m-%Y_%H-%M-%S")
        callbacks = ModelCheckpoint(model.name+f'_W_D{date}.hdf5', monitor='val_loss', verbose=verbose, save_best_only=True, mode='min')
    else:
        callbacks=None

    model_history = model.fit(train, epochs=num_epochs, batch_size=32, verbose=verbose,
                                validation_data=validation,
                                callbacks=callbacks)
    return model_history

def LoadModel(name=None, path_models='./models/'):
    """
    load diffrent types of models.
    name: name of the model.
        - if None, selects the newest model
        - if no weights are given, picks latest available.
    (please use the weights.hdf5 to refer to the model)

    currently supports:
        - old models trained pre-namechange
        - AE_v1_* : models of the new name sceme
    """
    if not(name):
        files = [path_models + x for x in os.listdir(path_models) if x.endswith(".hdf5")]
        name = max(files , key = os.path.getctime).replace(path_models,"")  # cut off the path_models, to remain with a name

    if name[:6] =='AE_v1_':
        # the model is of the first iteration
        i1 = name.find('_W_D') # check if this one has the wheights
        if i1 >=0:
            model_path  = path_models + name[:i1] + '.json'
            weight_path = path_models + name
        else: # this one contains no wheights, find latest wheights available
            name.replace('.json','') # remove json if it was present
            model_path  = path_models + name + '.json'

            files = [path_models + x for x in os.listdir(path_models) if x.endswith(".hdf5") and x.startswith(name)]
            weight_path = max(files , key = os.path.getctime)
    else:
        # the model is of the old type:
        name.replace('_w.hdf5','')
        name.replace('.json','')
        model_path  = name + ".json"
        weight_path = name + "_w.hdf5"

    #%% load model and model weights
    with open(model_path+'.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights(weight_path)

    return model


def get_pcam_generators(base_dir, train_batch_size=32, val_batch_size=32, classes =['0'], class_mode='input'):
     # dataset parameters
     train_path = os.path.join(base_dir, 'train+val', 'train')
     valid_path = os.path.join(base_dir, 'train+val', 'valid')

     RESCALING_FACTOR = 1./255

     # instantiate data generators
     datagen = ImageDataGenerator(rescale=RESCALING_FACTOR)

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
