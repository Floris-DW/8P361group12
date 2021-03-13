# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 14:37:29 2021

@author: 20183193
"""
"NOTE: curently we do not have a proper test set, so a subset of the validation set is used"

#%% imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import model_from_json

#%% define paths
path_images = '../../Images/' # navigate to ~/source/Images from ~/source/Github/autoencoder.py
path_models = './models/'

#model_path = path_models + r'09-03-2021_15-35-37_model_autoencoder_Ldim-2'

use_latest_model = True
if use_latest_model:
    files = [path_models + x for x in os.listdir(path_models) if x.endswith(".json")]
    model_path = max(files , key = os.path.getctime)[:-5]  # the [:-5] cuts off the .json


#%% define helper function.
# this helper function is now defined in both files.
# it may be beneficial to swich to a main() and import strategy later.
IMAGE_SIZE = 96
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

#%% load model and model weights
with open(model_path+'.json', 'r') as json_file:
    loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)
model.load_weights(model_path+'_w.hdf5')

#%% plot test:
n = 10

_, image_test = get_pcam_generators(path_images, train_batch_size=n, classes=['0'], class_mode=None)
images = image_test.next()

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

#%% Registration error

real_img = images[0] - np.mean(images[0])
recreated_img = decoded_imgs[0] - np.mean(decoded_imgs[0])

u = real_img.reshape((real_img.shape[0]*real_img.shape[1],3))
v = recreated_img.reshape((recreated_img.shape[0]*recreated_img.shape[1],3))

u = u - u.mean(keepdims=True)
v = v - v.mean(keepdims=True)

CC=(np.transpose(u).dot(v))/(((np.transpose(u).dot(u))**0.5)*((np.transpose(v).dot(v))**0.5))

print(np.mean(CC))
