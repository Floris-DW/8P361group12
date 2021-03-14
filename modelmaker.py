# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 20:23:46 2021

@author: 20183193
"""
"""
refrance source:
    https://github.com/seasonyc/densenet
"""
version = 'AE_v1' # naming sceme for models, prevent namingconflicts. AE = AutoEncoder

# imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D, Input
from tensorflow.keras.callbacks import ModelCheckpoint

import time
# Preparations
path_images = '../../Images/' # navigate to ~/source/Images from ~/source/Github/autoencoder.py
path_models = './models/'

def AutoEncoder(input_shape=(96,96,3), filters=[64, 32, 16], kernel_size=(3,3), pool_size=(2,2),
                activation='relu',padding='same', model_name=None):
    #encoder
    input_layer = Input(shape=input_shape)
    x = input_layer
    for i in filters:
        x = Conv2D(i, kernel_size, activation=activation, padding=padding)(x)
        x = MaxPool2D(pool_size=pool_size, padding=padding)(x)

    # decoder
    for i in filters[::-1]: #loop over the filter in reverse
        x = Conv2D(filters[2], kernel_size, activation=activation, padding=padding)(x)
        x = UpSampling2D(pool_size)(x)
    x = Conv2D(input_shape[2], kernel_size, activation="sigmoid", padding=padding)(x) # output layer

    # assign the model an default name if None was given
    r = lambda x: str(x).replace(', ','.')[1:-1] # remove the (,),[,] and replace , with .
    model_name = model_name or f"{version}_F{ r(filters) }_K{ r(kernel_size) }_P{ r(pool_size) }"

    return Model(input_layer, x, name=model_name)


def TrainModel(model, train, validation, num_epochs, batch_size=32,
               loss='MeanSquaredError', optimizer='adam',
               save_model=True, verbose=1, save_dir='./models/'):

    if save_model:
        model_json = model.to_json() # serialize model to JSON
        with open(save_dir + model.name+'.json', 'w') as json_file:
            json_file.write(model_json)

        # prepare to save wheights
        date = time.strftime("%d-%m-%Y_%H-%M-%S")
        callbacks = ModelCheckpoint(save_dir + model.name+f'_W_E{num_epochs}_D{date}.hdf5', monitor='val_loss', verbose=verbose, save_best_only=True, mode='min')
    else:
        callbacks=None

    model.compile(optimizer,loss)
    model_history = model.fit(train, epochs=num_epochs, batch_size=32, verbose=verbose,
                                validation_data=validation,
                                callbacks=callbacks)
    return model_history


def LoadModel(name='', path_models='./models/'):
    """
    load diffrent types of models.
    name: name of the model.
        - if '', selects the newest model
        - if no weights are given, picks latest available for the given name.

    **please use the weights (.hdf5) to refer to the model**
    """
    # if the given name does not contain the wheights or if no name given
    if not(name.endswith(".hdf5")):
        files = [path_models + x for x in os.listdir(path_models) if x.startswith(name) and x.endswith(".hdf5")]
        name = max(files , key = os.path.getctime).replace(path_models,'')

    i = name.upper().find("_W")
    model_path  = path_models + name[:i] + '.json'
    weight_path = path_models + name

    # load model and model weights
    with open(model_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights(weight_path)
    return model


def ImageGeneratorsTrain(base_dir, train_batch_size=32, val_batch_size=32, IMAGE_SIZE=96, split=0.3):
     """
     create twe genarators:
     train and validation.
     the train and validaion set are build from the split training set.
     """
     # dataset parameters
     train_path = base_dir + 'train+val/train'

     RESCALING_FACTOR = 1./255

     # we split the training set into training and validation.
     datagen_train = ImageDataGenerator(rescale=RESCALING_FACTOR, validation_split=split)

     # using the method from here might be better:
     # https://towardsdatascience.com/addressing-the-difference-between-keras-validation-split-and-sklearn-s-train-test-split-a3fb803b733

     train_gen = datagen_train.flow_from_directory(train_path, batch_size=train_batch_size, target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             subset='training',   classes='0', class_mode='input')

     val_gen = datagen_train.flow_from_directory(train_path, batch_size=val_batch_size, target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             subset='validation', classes='0', class_mode='input')
     return train_gen, val_gen


def ImageGeneratorsTest(base_dir, batch_size=32, IMAGE_SIZE=96):
     """
     create two genarators:
     test_healthy, test_diseased
     the test sets are the original validaion set.
     """
     # dataset parameters
     valid_path = base_dir + 'train+val/valid'

     RESCALING_FACTOR = 1./255

     #The test set is made from the validaion set
     datagen_test = ImageDataGenerator(rescale=RESCALING_FACTOR)

     # the test generators
     test_gen_H = datagen_test.flow_from_directory(valid_path, batch_size=batch_size, target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             classes='0', class_mode=None)

     test_gen_D = datagen_test.flow_from_directory(valid_path, batch_size=batch_size, target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             classes='1', class_mode=None)

     return test_gen_H, test_gen_D


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    if True:
        model = AutoEncoder()
        num_epochs = 10
        train_gen, val_gen = ImageGeneratorsTrain(path_images)
        history = TrainModel(model, train_gen, val_gen, num_epochs)
    else:
        model = LoadModel()
    model.summary()

    # visualize:
    test_gen_H, test_gen_D = ImageGeneratorsTest(path_images)

    n = 10

    images = test_gen_H.next()

    decoded_imgs =  model.predict(images)

    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(images[i])
        #plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i+1+n)
        plt.imshow(decoded_imgs[i])
        #plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()